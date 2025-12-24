# imports
import os
import sys
import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out
import sys
import h5py
import csv

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from cddd_main import InferenceModel
from similarity import ChemblNearestNeighbour

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read input
_, smiles_list = read_smiles(input_file)

smiles_indexed =[]
with open(os.path.join(root, "..", "..", "checkpoints","fpsim2_database_chembl_smiles.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        smiles_indexed += [r[0]]

# run model
model = InferenceModel()
outputs = model.seq_to_emb(smiles_list)

nan_rows = np.isnan(outputs).any(axis=1)
n_rows_with_nan = int(nan_rows.sum())
if n_rows_with_nan != 0:
    chembl_sim  = ChemblNearestNeighbour()
    for i in np.where(nan_rows)[0]:
        smi = smiles_list[i]
        nn_idx = chembl_sim.highest_similarity(smi)
        if nn_idx is None:
            continue
        smiles_chembl = smiles_indexed[nn_idx]
        print(smi, smiles_chembl)
        chembl_value = model.seq_to_emb([smiles_chembl])[0]
        outputs[i, :] = chembl_value
    nan_rows_after = np.isnan(outputs).any(axis=1)

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

num_dims = outputs.shape[1]
header = [f"cddd_{str(i).zfill(3)}" for i in range(num_dims)]

# write output in a .csv file
write_out(outputs, header, output_file, np.float32)
