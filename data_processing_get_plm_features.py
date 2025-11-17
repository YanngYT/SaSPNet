import os
import torch
from utils import *

data_dir = "./test_data_processed"
seq_txt_path = os.path.join(data_dir, "data_list.txt")
save_pt_path = os.path.join(data_dir, "ESM_features.pt)

torch.cuda.set_device(1)

def sequences_txt_to_esm_pt():

    os.makedirs(os.path.dirname(save_pt_path), exist_ok=True)

    raw_data = []
    with open(seq_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s[:70]
            raw_data.append(("protein", s))

    if len(raw_data) == 0:
        raise RuntimeError(f"no valid sequences found in {seq_txt_path}")

    split = 100
    total = len(raw_data)
    results = []

    full_batches = total // split
    for i in range(full_batches):
        batch = raw_data[i * split : (i + 1) * split]
        emb = trans_data_esm(batch)
        results.append(emb)

    if total % split != 0:
        batch = raw_data[full_batches * split : total]
        emb = trans_data_esm(batch)
        results.append(emb)

    feat = torch.cat(results, dim=0)
    if feat.dim() != 2 or feat.size(1) != 1280:
        raise RuntimeError(f"ESM features got wrong shape: {tuple(feat.size())}, expected (N, 1280)")

    torch.save(feat.cpu(), save_pt_path)
    print(f"PLM features saved at: {save_pt_path}")
    print(f"shape: {tuple(feat.size())}")

if __name__ == "__main__":
    sequences_txt_to_esm_pt()
