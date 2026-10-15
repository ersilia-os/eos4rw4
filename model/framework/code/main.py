# imports
import csv
import os
import sys
import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from cddd_main import InferenceModel

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read input
_, smiles_list = read_smiles(input_file)

# run model
model = InferenceModel()
outputs = model.seq_to_emb(smiles_list)

# a molecule that cannot be embedded (invalid SMILES, outside the encoder's
# applicability domain, or any other reason) gets a fully empty row
no_embedding = ~np.isfinite(outputs).all(axis=1)
outputs[no_embedding, :] = np.nan

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

num_dims = outputs.shape[1]
header = [f"cddd_{str(i).zfill(3)}" for i in range(num_dims)]


def write_csv_empty_missing(results, header, file):
    """Write a .csv file in which missing values are empty cells, not "nan"."""
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row, missing in zip(results, no_embedding):
            writer.writerow([""] * len(header) if missing else row)


# write output in a .csv file
if output_file.endswith(".csv"):
    write_csv_empty_missing(outputs, header, output_file)
else:
    write_out(outputs, header, output_file, np.float32)
