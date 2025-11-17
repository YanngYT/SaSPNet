import os

fasta_path = "./data/test_set.fasta"
seq_out_path = "./test_data_processed/data_list.txt"
kingdom_out_path = "./test_data_processed/kingdom_list.txt"
target_out_path = "./test_data_processed/target_list.txt"
aa_out_path = "./test_data_processed/aa_list.txt"

def parse_fasta_and_save():
    os.makedirs(os.path.dirname(seq_out_path), exist_ok=True)
    with open(fasta_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"FASTA format error")

    seq_list = []
    kingdom_list = []
    target_list = []
    label_list = []

    for i in range(0, len(lines), 3):
        header = lines[i]
        sequence = lines[i + 1]
        label = lines[i + 2]

        if not header.startswith('>'):
            raise ValueError(f"row {i+1} has invalid header: {header}")

        parts = header[1:].split('|')

        if len(parts) == 3:
            sample_id, kingdom, sp_type = parts
        elif len(parts) == 4:
            sample_id, kingdom, sp_type, _ = parts
        else:
            print(f"warning: sample {i//3+1} has abnormal header: {header}")
            continue

        seq_list.append(sequence)
        kingdom_list.append(kingdom)
        target_list.append(sp_type)
        label_list.append(label)
    with open(seq_out_path, 'w') as f:
        f.write("\n".join(seq_list) + "\n")
    with open(kingdom_out_path, 'w') as f:
        f.write("\n".join(kingdom_list) + "\n")
    with open(target_out_path, 'w') as f:
        f.write("\n".join(target_list) + "\n")
    with open(aa_out_path, 'w') as f:
        f.write("\n".join(label_list) + "\n")

if __name__ == "__main__":
    parse_fasta_and_save()
