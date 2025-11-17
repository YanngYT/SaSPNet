import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GCNConv, global_mean_pool
from utils import *

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, device=torch.device('cuda:1')):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(input_dim, output_dim)
        self.device = device

    def forward(self, x, adj_matrix):
        x = x.to(self.device)
        edge_index = self.adj_to_edge_index(adj_matrix).to(self.device)
        h = self.conv(x, edge_index).to(self.device)
        h = F.relu(h)
        return h

    def adj_to_edge_index(self, adj_matrix):
        # adj_matrix  [batch_size, num_nodes, num_nodes]
        batch_size, num_nodes, _ = adj_matrix.shape
        edge_list = []
        for i in range(batch_size):
            adj = adj_matrix[i]
            edge_indices = torch.nonzero(adj, as_tuple=False)
            edge_list.append(edge_indices.t())

        edge_index = torch.cat(edge_list, dim=1)  # [2, num_edges]
        return edge_index

# structure encoder implementation
class GraphEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, device=torch.device('cuda:1')):
        super(GraphEncoder, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, device) 
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, device) 
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim, device) 

    def forward(self, node_feature_matrix, adj_matrix):
        h1 = self.gcn1(node_feature_matrix, adj_matrix) 
        h2 = self.gcn2(h1, adj_matrix) 
        h3 = self.gcn3(h2, adj_matrix)
        h = torch.cat((h1, h2, h3), dim=-1)  # (batch_size, 70, hidden_dim * 3)

        return h

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TextCNN, self).__init__()
        self.activation = nn.ReLU()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size, padding=math.floor(kernel_size/2)),
                                    self.activation)

    def forward(self, x):
        embed_x = x
        embed_x = embed_x.permute(0, 2, 1)
        out = self.conv(embed_x.float())
        out=out.permute(0 ,2, 1)
        return out

class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device, max_text_len=70, dropout_rate=0.5):
        super(BLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2
        self.dropout_rate = dropout_rate
        self.max_text_len = max_text_len
        self.device = device

        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_directions, self.hidden_dim * self.num_directions),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout_rate,
                               batch_first=True,
                               bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(self.num_directions * self.hidden_dim)
        self.start_dim = int(self.num_directions * self.hidden_dim * self.max_text_len )

    def forward(self, input):
        batch_size = input.shape[0]
        input = input.float().to(self.device)
        hidden_state = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim).to(self.device)
        cell_state = torch.randn(self.num_layers*self.num_directions, batch_size, self.hidden_dim).to(self.device)
        self.lstm.flatten_parameters()
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        # outputs shape: (batch_size, 70, num_direction*hidden_dim)
        outputs = outputs.permute(1, 2, 0) 
        outputs = self.batch_norm(outputs)
        outputs = outputs.permute(2, 0, 1) 
        model = outputs
        return model

class Multihead_Attention(nn.Module):
    def __init__(self, num_units, num_heads=1, dropout_rate=0.5, gpu=True, causality=False):
        super(Multihead_Attention, self).__init__()
        self.gpu = gpu
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        if self.gpu:
            self.Q_proj = self.Q_proj.cuda()
            self.K_proj = self.K_proj.cuda()
            self.V_proj = self.V_proj.cuda()
        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)
        outputs = outputs / (K_.size()[-1] ** 0.5)
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)
        if last_layer == True:
            return outputs
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        outputs += queries

        return outputs

class LSTM_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, d_model, num_heads, device, max_text_len=70, dropout_rate=0.5):
        super(LSTM_attention, self).__init__()
        self.device = device
        self.lstm = BLSTM(input_dim, hidden_dim, num_layers, self.device, max_text_len, dropout_rate)
        self.label_attn = Multihead_Attention(d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        self.linear = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.droplstm = nn.Dropout(dropout_rate)

    def forward(self,lstm_out,label_embs):
        lstm_out = self.lstm(lstm_out)
        lstm_out = self.droplstm(lstm_out)
        # lstm_out (seq_length * batch_size * hidden)
        label_attention_output = self.label_attn(lstm_out, label_embs, label_embs)
        # label_attention_output (batch_size, seq_len, embed_size)
        lstm_out = torch.cat([lstm_out, label_attention_output], -1)
        return lstm_out


class SaSPNet(nn.Module):
    def __init__(self, device):
        super(SaSPNet, self).__init__()
        self.sequence_len = 70  # max length of SP sequence
        self.vocab_size = 21  
        self.embedding_dim = 20 
        self.seq_feature_dim = 256  
        self.d_model = 128  
        self.struc_feature_dim = 512  
        self.plm_feature_dim = 1280  
        self.sp_classes = 6 
        self.site_classes = 11
        self.device = device

        # embedding layer
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # sequence encoder
        self.cnn_block = TextCNN(self.embedding_dim + 4, self.seq_feature_dim, 3)
        self.bilstm_block = LSTM_attention(self.embedding_dim + 4, self.seq_feature_dim//4,2, self.d_model, 8, self.device, self.sequence_len, 0.5)
        self.transition = nn.Linear(self.embedding_dim + 4, self.d_model)

        # structure encoder
        self.gnn = GraphEncoder(input_dim=self.seq_feature_dim, hidden_dim=self.struc_feature_dim, device=device)

        # prediction heads
        self.type_prediction_heads = nn.Sequential(
            nn.Linear(self.seq_feature_dim + self.plm_feature_dim + self.struc_feature_dim * 3, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, self.sp_classes)
        )
        self.site_prediction_heads = nn.Sequential(
            nn.Linear(self.seq_feature_dim + self.struc_feature_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.site_classes)
        )

    def forward(self, seq, struc, plm_features):
        # sequence data processing
        seq_indices = seq[:, :self.sequence_len].long().to(self.device) # shape: (batch_size, 70)
        group_info = seq[:, self.sequence_len:self.sequence_len + 4].unsqueeze(1).expand(-1, self.sequence_len, -1)
        group_info = group_info.long().to(self.device) # shape: (batch_size, 70, 4)
        plm_features = plm_features.to(torch.float).to(self.device)
	
	    # embedding layer
        seq_embedding = self.embedding_layer(seq_indices).to(self.device)  # output shape: (batch_size, 70, embedding_dim=20)
        seq_embedding = torch.cat((seq_embedding, group_info), dim=2).to(self.device)  # shape: (batch_size, 70, 24)

        # sequence encoder
        cnn_features = self.cnn_block(seq_embedding).to(self.device)  # shape: (batch_size, 70, seq_feature_dim=256)
        label_embs = self.transition(seq_embedding).to(self.device) # shape: (batch_size, 70, d_model=128)
        bilstm_features = self.bilstm_block(seq_embedding, label_embs).to(self.device)  # shape: (batch_size, 70, seq_feature_dim=256)
        seq_features = cnn_features + bilstm_features  # shape: (batch_size, 70, seq_feature_dim=256)

        # structure encoder
        struc_features = self.gnn(seq_features, struc).to(self.device) # shape: (batch_size, 70, struc_feature_dim * 3)

        # features for type prediction
        type_features = torch.cat((seq_features, struc_features), dim=-1).to(self.device)  # shape: (batch_size, 70, seq_feature_dim + struc_feature_dim * 3)
        type_features = global_mean_pool(type_features, batch=None).to(self.device)
        type_features = torch.cat((type_features, plm_features), dim=-1).to(self.device)  # shape: (batch_size, seq_feature_dim + struc_feature_dim * 3 + plm_feature_dim)
        type_logit = self.type_prediction_heads(type_features).to(self.device) # (batch_size, sp_classes)

        # features for cleavage site prediction
        site_features = torch.cat((seq_features, struc_features), dim=-1).to(self.device)
        site_features = site_features.reshape(-1, self.seq_feature_dim + self.struc_feature_dim * 3)  # (batch_size * sequence_len, seq_feature_dim + struc_features * 3)
        site_logit = self.site_prediction_heads(site_features).to(self.device)  # (batch_size * sequence_len, site_classes)
        batch_size = type_logit.shape[0]
        site_logit = site_logit.reshape(batch_size, self.sequence_len, self.site_classes) # (batch_size, sequence_len, site_classes)

        return type_logit, site_logit

    # model_predict implements the final prediction
    def model_predict(self, seq, struc, plm_features):
        type_logit, site_logit = self.forward(seq, struc, plm_features)
        type_prob = F.softmax(type_logit, dim=-1).to(self.device)  # (batch_size, sp_classes)
        return type_prob, site_logit


class SaSPNetLoss(nn.Module):
    def __init__(self, device, scale_factor=1.0, tau=1.0, sp_classes=6, site_classes=11):
        super(SaSPNetLoss, self).__init__()
        self.sp_classes = sp_classes
        self.site_classes = site_classes
        self.tau = tau
        self.scale_factor = scale_factor
        self.device = device

    def forward(self, type_logits, site_logits, type_labels, site_labels):
        sp_loss = self.signal_peptide_loss(type_logits, type_labels).to(self.device)
        site_loss = self.cleavage_site_loss(site_logits, site_labels).to(self.device)
        total_loss = sp_loss + self.tau * site_loss
        return total_loss

    def signal_peptide_loss(self, logits, labels):
        # Δy
        n_j = torch.bincount(labels, minlength=self.sp_classes).float()
        #  C
        C = self.compute_C(labels, self.sp_classes)
        # margin
        margin = C / (n_j ** 0.25)
        margin_per_sample = margin[labels].unsqueeze(1).to(self.device)  # 形状 (batch_size, 1)

        logits = (logits - margin_per_sample).to(self.device)
        labels = labels.to(self.device)

        loss = F.cross_entropy(self.scale_factor * logits, labels, reduction='mean').to(self.device)
        return loss

    def cleavage_site_loss(self, logits, labels):
        batch_size, seq_len, site_classes = logits.shape
        logits_flat = logits.view(-1, site_classes)  # (batch_size * seq_len, site_classes)
        labels_flat = labels.view(-1)  # (batch_size * seq_len)

        n_j = torch.bincount(labels_flat, minlength=self.site_classes).float()
        C = self.compute_C(labels_flat, self.site_classes)
        margin = C / (n_j ** 0.25 + 1e-6)  
        margin_per_sample = margin[labels_flat].unsqueeze(1).to(self.device)

        logits_margin = (logits_flat - margin_per_sample).to(self.device)
        labels_flat = labels_flat.to(self.device)

        loss = F.cross_entropy(self.scale_factor * logits_margin, labels_flat, reduction='mean')
        return loss

    def compute_C(self, labels, num_classes=None):
        if num_classes is None:
            num_classes = self.sp_classes
        n_j = torch.bincount(labels, minlength=num_classes).float()
        n_j_inv_4 = n_j ** (-1 / 4)
        C = 0.5 / n_j_inv_4.max()
        return C


if __name__ == '__main__':
    device = torch.device("cuda:1")
    batch_size = 10
    seq = torch.randint(low=0, high=21, size=(batch_size, 74)).to(device)
    struc = torch.randint(low=0, high=2, size=(batch_size, 70, 70)).to(device)
    plm_features = torch.randn(size=(batch_size, 1280)).to(device)
    model = SaSPNet(device).to(device)
    type_prob, site_logit = model.model_predict(seq, struc, plm_features)
    type = type_prob.argmax(dim=-1)
    print(type)
    type_logit, site_logit = model(seq, struc, plm_features)
    type_labels = torch.randint(0, 6, (batch_size,))
    site_labels = torch.randint(0, 11, (batch_size, 70))
    criterion = SaSPNetLoss(device, scale_factor=1).to(device)
    loss = criterion(type_logit, site_logit, type_labels, site_labels).to(device)
    print(loss)
