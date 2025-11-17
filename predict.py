import torch as torch
import numpy as np
import pandas as pd
import os
from utils import *
from model import SaSPNet


device = torch.device("cuda:1")

data_dir = "./test_data_processed"

filename_list = ["data_list.txt",
                 "kingdom_list.txt",
                 "ESM_features.pt",
                 "ESMFold_adj_matrices",
                 "target_list.txt",
                 "aa_list.txt"
                 ]
for i in range(len(filename_list)):
    filename_list[i] = os.path.join(data_dir, filename_list[i])

save_path = "results.csv"


def createEvalData(data_path, kingdom_path, PLM_feature_path, struc_path, label_path, site_path) :
    data_list = []
    kingdom_list = []
    label_list = []
    maxlen = 70
    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = np.array(trans_data(line.strip('\n')[0:70], maxlen))
            data_list.append(str)
    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            kingdom_list.append(np.eye(len(kingdom_dic.keys()))[kingdom_dic[line.strip('\n\t')]])

    if os.path.exists(PLM_feature_path):
        plm_feature = torch.load(PLM_feature_path)
    else:
        plm_feature = None

    with open(label_path, 'r') as label_file:
        for line in label_file:
            str = np.array(trans_label(line.strip('\n')[0:70]))
            label_list.append(str)

    data = np.array(data_list)
    kingdoms = np.array(kingdom_list)
    labels = np.array(label_list)
    seq = np.concatenate((data, kingdoms), axis=1)
    seq = torch.tensor(seq)

    struc = load_adj_matrices(struc_path)

    labels = torch.tensor(labels)

    aa_list = []
    with open(site_path, 'r') as aa_file:
        for line in aa_file:
            aa_list.append(classes_sequence_from_ann_sequence(line.strip("\n\t")))
    aas = np.array(aa_list)
    site_labels = torch.tensor(aas)

    return seq, struc, plm_feature, labels, site_labels

def trans_output(idx):
    return dic2[int(idx)]



if __name__ == '__main__':
    model_path = "./model_pth/SaSPNet.pth"
    model = SaSPNet(device)
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    seq, struc, plm_feature, labels, site_labels = createEvalData(data_path=filename_list[0],
                            kingdom_path=filename_list[1],
                            PLM_feature_path=filename_list[2],
                            struc_path=filename_list[3],
                            label_path=filename_list[4],
                            site_path=filename_list[5])
    test_data = torch.utils.data.TensorDataset(seq, struc, plm_feature, labels, site_labels)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    
    predicted_types = []
    predicted_cleavages = []
    raw_sequences = []

    with open(filename_list[0], "r") as f:
        for line in f:
            raw_sequences.append(line.strip("\n"))
    seq_index = 0

    with torch.no_grad():
        for seq_inputs, struc_inputs, plm_inputs, label, site_label in test_loader:
            seq_inputs = seq_inputs.to(device)
            struc_inputs = struc_inputs.to(device)
            plm_inputs = plm_inputs.to(device)
            output, site_logits = model.model_predict(seq_inputs, struc_inputs, plm_inputs)

            preds = output.argmax(dim=-1).cpu().numpy()
            site_argmax = torch.argmax(site_logits.cpu(), dim=2).numpy()
            for i in range(len(preds)):
                predicted_types.append(trans_output(preds[i]))
                site_mask = site_argmax[i].copy()
                site_mask[site_mask == 1] = 100
                site_mask[np.isin(site_mask, [0, 3])] = 1
                site_mask[site_mask != 1] = 0
                seq_string = raw_sequences[seq_index]
                if preds[i] == 0:
                    predicted_cleavages.append("")
                else:
                    idx = np.where(site_mask == 1)[0]
                    if len(idx) == 0:
                        predicted_cleavages.append(seq_string)
                    else:
                        cut = idx[0]
                        predicted_cleavages.append(seq_string[:cut + 1])

                seq_index += 1

    df = pd.DataFrame({
        "sequence": raw_sequences,
        "predicted_type": predicted_types,
        "predicted_cleavage": predicted_cleavages,
    })
    df.to_csv(save_path, index=False)
    print("Prediction saved to:", save_path)
