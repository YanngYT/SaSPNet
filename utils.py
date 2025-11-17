import torch
import os
from Bio.PDB import PDBParser
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, cohen_kappa_score, classification_report, f1_score, confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
import esm
from enum import Enum

def extract_ca_coordinates(pdb_file, target_len=70):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':
                    continue
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
                else:
                    ca_coords.append(np.zeros(3))
    ca_coords = np.array(ca_coords)
    L = ca_coords.shape[0]
    if L < target_len:
        pad = np.zeros((target_len - L, 3))
        ca_coords = np.vstack([ca_coords, pad])
    else:
        ca_coords = ca_coords[:target_len]
    return ca_coords

def pdb_to_adj_matrix(pdb_file, target_num_nodes=70, radius=10.0):
    ca_coords = extract_ca_coordinates(pdb_file, target_len=target_num_nodes)  # [70,3]
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]  # [70,70,3]
    dist_matrix = np.linalg.norm(diff, axis=-1)  # [70,70]
    adj = (dist_matrix < radius).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj  # [70,70]

def process_pdbs_to_adj_matrices(pdb_path, adj_path, target_num_nodes=70, radius=10.0):
    os.makedirs(adj_path, exist_ok=True)
    i = 1
    while True:
        pdb_name = f"peptide_{i}.pdb"
        pdb_file = os.path.join(pdb_path, pdb_name)
        if not os.path.isfile(pdb_file):
            break
        try:
            adj = pdb_to_adj_matrix(pdb_file, target_num_nodes=target_num_nodes, radius=radius)
            save_name = f"adj_matrix_{i}.npy"
            save_path = os.path.join(adj_path, save_name)
            np.save(save_path, adj)
            print(f"[{i}] Saved adjacency matrix to {save_name}")
        except Exception as e:
            print(f"[Error] {pdb_name} failed: {e}")
        i += 1
    if i == 1:
        print("warning：no peptide_i.pdb files found")

dic = {'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3, 'TATLIPO' : 4, 'PILIN' : 5}
dic2 = {0: 'NO_SP', 1: 'SP', 2: 'LIPO', 3: 'TAT', 4: 'TATLIPO', 5: 'PILIN'}
kingdom_dic = {'EUKARYA':0, 'ARCHAEA':1, 'POSITIVE':2, 'NEGATIVE': 3}

def trans_data_esm(str_array):
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = (esm_model).cuda()
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(str_array)
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33].cuda()
    sequence_representations = []
    for i, (_, seq) in enumerate(str_array):
        temp_tensor = token_representations[i, 1: len(seq) + 1]
        sequence_representations.append(temp_tensor.mean(0).detach().cpu().numpy())
    result = torch.tensor(np.array(sequence_representations))
    return result

def trans_data(str1, padding_length):
    a = []
    trans_dic = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':0}
    for i in range(len(str1)):
        if (str1[i] in trans_dic.keys()):
            a.append(trans_dic.get(str1[i]))
        else:
            print("Unknown letter:" + str(str1[i]))
            a.append(trans_dic.get('X'))
    while(len(a)<padding_length):
        a.append(0)

    return a

def trans_label(str1):
    if((str1) in dic.keys()):
        a = dic.get(str1)
    else:
        print(str1)
        raise Exception('Unknown category!')

    return a

def load_adj_matrices(load_path, strict=True):
    files = [f for f in os.listdir(load_path) if f.startswith("adj_matrix_") and f.endswith(".npy")]
    indices = []
    for f in files:
        try:
            idx_str = f[len("adj_matrix_"):-len(".npy")]
            idx = int(idx_str)
            indices.append(idx)
        except ValueError:
            continue
    if not indices:
        raise RuntimeError(f"no adj_matrix_i.npy files found")
    max_idx = max(indices)
    adj_list = []
    count = 0
    for i in range(1, max_idx + 1):
        file_name = f"adj_matrix_{i}.npy"
        file_path = os.path.join(load_path, file_name)
        if not os.path.isfile(file_path):
            msg = f"{file_name} doesn't exist"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print(f"warning: {msg}")
                continue
        adj_matrix = np.load(file_path)
        if adj_matrix.shape != (70, 70):
            print(f"warning: shape of {file_name} is {adj_matrix.shape}, expected (70, 70)")
        adj_list.append(adj_matrix)
        count += 1
        print(f"loaded {count}: {file_name}")
    if len(adj_list) == 0:
        raise RuntimeError("no adjacency matrix loaded")
    adj_batch = torch.tensor(np.array(adj_list), dtype=torch.float32)  # [batch_size, 70, 70]
    return adj_batch

class AnnotationLetter(Enum):
    INNER = "I"
    OUTER = "O"
    TRANSMEMBRANE = "M"
    # Includes cleavage site AA
    SIGNAL_SEC_SP1 = "S"
    # Includes cleavage site AA
    SIGNAL_SEC_SP2 = "L"
    # Includes cleavage site AA
    SIGNAL_TAT_SP1 = "T"
    SIGNAL_SEC_SP3 = "P"

class PositionSpecificLetter(Enum):
    # 9
    SIGNAL_SEC_SP3 = "P"
    # 6
    SIGNAL_SEC_SP1 = "S"
    # 8
    SIGNAL_SEC_SP2 = "Z"
    # 7
    SIGNAL_TAT_SP1 = "T"
    # 0
    CLEAVAGE_SITE_SP1 = "C"
    # 3
    CLEAVAGE_SITE_SP2 = "K"
    # 5
    OUTER = "O"
    # 2
    INNER = "I"
    # 4
    TRANSMEMBRANE_IN_OUT = "L"
    # 1
    TRANSMEMBRANE_OUT_IN = "E"
    #10
    EMPTY = "z"

    @classmethod
    def values(cls):
        return [e.value for e in cls]

def classes_sequence_from_ann_sequence(sequence):
    position_specific_classes_enc = preprocessing.LabelEncoder()
    position_specific_classes_enc.fit(
        np.array(PositionSpecificLetter.values()).reshape((len(PositionSpecificLetter.values()), 1))
    )
    classes_sequence = []
    prev_inner_outer = None
    for i in range(70):
        if (len(sequence) < i + 1):

            letter = None
        else:
            letter = sequence[i]
        if letter is None:

            position_specific_class = PositionSpecificLetter.EMPTY

            transformed = position_specific_classes_enc.transform([position_specific_class.value])
            classes_sequence.append(transformed)

            continue

        prev_letter = AnnotationLetter(sequence[i - 1]) if i > 0 else None
        next_letter = AnnotationLetter(sequence[i + 1]) if i + 1 < len(sequence) else None

        position_specific_class = None
        letter = AnnotationLetter(letter)

        if letter == AnnotationLetter.INNER:
            position_specific_class = PositionSpecificLetter.INNER
            prev_inner_outer = AnnotationLetter.INNER
        elif letter == AnnotationLetter.OUTER:
            position_specific_class = PositionSpecificLetter.OUTER
            prev_inner_outer = AnnotationLetter.OUTER
        elif letter == AnnotationLetter.TRANSMEMBRANE:
            if prev_letter == AnnotationLetter.TRANSMEMBRANE:
                if prev_inner_outer == AnnotationLetter.INNER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
                elif prev_inner_outer == AnnotationLetter.OUTER:
                    position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
            elif (
                prev_letter == AnnotationLetter.INNER or prev_inner_outer == AnnotationLetter.INNER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_IN_OUT
            elif (
                prev_letter == AnnotationLetter.OUTER or prev_inner_outer == AnnotationLetter.OUTER
            ):
                position_specific_class = PositionSpecificLetter.TRANSMEMBRANE_OUT_IN
        elif letter == AnnotationLetter.SIGNAL_SEC_SP1:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP2:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP2:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP2
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP2
        elif letter == AnnotationLetter.SIGNAL_TAT_SP1:

            if next_letter == AnnotationLetter.SIGNAL_TAT_SP1:
                position_specific_class = PositionSpecificLetter.SIGNAL_TAT_SP1
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1
        elif letter == AnnotationLetter.SIGNAL_SEC_SP3:
            if next_letter == AnnotationLetter.SIGNAL_SEC_SP3:
                position_specific_class = PositionSpecificLetter.SIGNAL_SEC_SP3
            else:
                position_specific_class = PositionSpecificLetter.CLEAVAGE_SITE_SP1

        if position_specific_class is None:
            print("Unexpected case", prev_letter, letter, next_letter)

        transformed = position_specific_classes_enc.transform([position_specific_class.value])
        classes_sequence.append(transformed)

    seq_tensor = np.array(classes_sequence).reshape((70))

    return seq_tensor

def compute_metrics_on_minority(output_list, prediction_list, label_list, num_classes=6, minority_classes=[2, 3, 4, 5]):
    prediction_array = np.array(prediction_list)  # shape: (N,)
    label_array = np.array(label_list)  # shape: (N,)

    mcc = matthews_corrcoef(label_array, prediction_array)
    
    minority_precision = precision_score(label_array, prediction_array, labels=minority_classes, average='macro', zero_division=0)
    minority_recall = recall_score(label_array, prediction_array, labels=minority_classes, average='macro', zero_division=0)
    minority_f1 = f1_score(label_array, prediction_array, labels=minority_classes, average='macro', zero_division=0)

    return {
        "MCC": mcc,
        "Minority_Precision": minority_precision,
        "Minority_Recall": minority_recall,
        "Minority_F1": minority_f1
    }

def evaluate_cleavage_site(site_logits_list, site_labels_list, label_list, SP_classes=[0, 1, 2, 3, 4, 5], device=torch.device("cpu")):
    site_logits_tensor = torch.cat(site_logits_list, dim=0).to(device)  # [N, 70, 11]
    site_labels_tensor = torch.cat(site_labels_list, dim=0).to(device)  # [N, 70]
    label_list = [torch.tensor([x], device=device) if not isinstance(x, torch.Tensor) else x for x in label_list]
    label_tensor = torch.cat(label_list, dim=0)
    mask = torch.isin(label_tensor, torch.tensor(SP_classes, device=device))  # [N]
    selected_logits = site_logits_tensor[mask]  # [n, 70, 11]
    selected_labels = site_labels_tensor[mask]  # [n, 70]
    if selected_logits.shape[0] == 0:
        print(f"No samples found for SP_classes {SP_classes}.")
        return {
            'cs_acc': 0.0,
            'non_cs_acc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    output_aa = torch.argmax(selected_logits, dim=-1)  # shape: [n, 70]
    labels_test_aa = selected_labels  # shape: [n, 70]

    cleavage_classes = [0, 3]  # CLEAVAGE_SITE_SP1 (C), CLEAVAGE_SITE_SP2 (K)
    idx_cut_pred = torch.isin(output_aa, torch.tensor(cleavage_classes, device=device))
    output_aa_bin = torch.zeros_like(output_aa)
    output_aa_bin[idx_cut_pred] = 1 

    labels_test_aa_np = labels_test_aa.cpu().numpy()
    idx_cut_true = np.isin(labels_test_aa_np, cleavage_classes)
    labels_test_aa_bin = np.zeros_like(labels_test_aa_np)
    labels_test_aa_bin[idx_cut_true] = 1

    y_pred_aa = output_aa_bin.cpu()
    recall = metric_advanced('recall', y_pred_aa, labels_test_aa_bin)
    precision = metric_advanced('precision', y_pred_aa, labels_test_aa_bin)
    F1 = metric_advanced('F1_score', y_pred_aa, labels_test_aa_bin)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': F1
    }