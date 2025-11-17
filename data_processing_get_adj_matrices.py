from utils import *

if __name__ == '__main__':
    pdb_dir = "./test_data_processed/ESMFold_PDB"
    adj_dir = "./test_data_processed/ESMFold_adj_matrices"
    process_pdbs_to_adj_matrices(pdb_dir, adj_dir)