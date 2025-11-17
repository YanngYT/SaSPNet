# SaSPNet
This repository contains data and source code for the paper **Structure-Aware Multi-Modal Learning Improves Minor-Class Signal Peptide Prediction**.

## Repository Contents

### 1. Datasets

All datasets are provided in the `data` directory in FASTA format.

- `train_set.fasta`: USPNet training set
- `test_set.fasta`: USPNet benchmark test set
- `SP-MinorEval.fasta`: Our independent test set designed for minor-class signal peptides

### 2. Source code

A complete implementation of SaSPNet is provided in `model.py`.

### 3. Model Parameters

The trained SaSPNet model weights (`.pth` files) are stored in the `model_pth` directory.

### 4. Conda Environment

We provide `environment.yml`, which contains all required Python packages.
 You can rebuild the environment with:

```
conda env create -f environment.yml
```



## How to Use the Contents for Signal Peptide Prediction

### Step 1: Prepare your FASTA file

Place your input FASTA file in an appropriate directory.

### Step 2: Convert FASTA into txt files

Run:

```
python data_processing_fasta_to_txt.py
```

Before running, modify the paths in the script:

- `fasta_path`: path to your FASTA file
- `data_dir`: output directory for the processed txt files (e.g., `./test_data_processed`)

This script generates four txt files in the specified folder:

- `data_list.txt`
- `kingdom_list.txt`
- `target_list.txt`
- `aa_list.txt`

### Step 3: Generate ESM-2 features

Run:

```
python data_processing_get_plm_features.py
```

Remember to modify `data_dir` in this script to the directory containing the txt files before running.
The extracted ESM-2 features will be saved as a `.pt` file in the same directory.

### Step 4: Generate 3D structures using ESMFold or other tools

Use structure prediction models (e.g., ESMFold).
Platforms such as [OpenProtein.AI](https://www.openprotein.ai) can be used to obtain PDB files.

Rename each PDB file according to the order of sequences in the FASTA file:

```
peptide_1.pdb
peptide_2.pdb
...
```

Place all PDB files in a single directory.

### Step 5: Obtain adjacency matrices from PDB structures

Run:

```
python data_processing_get_adj_matrices.py
```

Before running, modify the script:

- `pdb_dir`: directory containing your PDB files
- `adj_dir`: output directory for adjacency matrices (recommended: a subfolder of the txt directory, e.g., `./test_data_processed/ESMFold_adj_matrices`)

The script generates adjacency matrices in `.npy` format, such as:

```
adj_matrix_1.npy
```

At this point, all preprocessing steps are completed.

### Step 6: Run SaSPNet to obtain predictions

Run:

```
python predict.py
```

Modify these fields inside the script before running:

- `data_dir`: directory containing the processed txt files
- `filename_list`: names of the required txt files, ESM-2 feature file, and the adjacency matrix directory
- `save_path`: output directory for prediction results
- `device`: computation device (e.g., `torch.device("cuda:1")`)

The predictions (signal peptide type and cleavage sites) will be saved in CSV format.
