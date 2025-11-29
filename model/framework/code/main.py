# imports
import os
import csv
import sys

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
from cddd_main import InferenceModel

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
model = InferenceModel()
outputs = model.seq_to_emb(smiles_list)

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

num_dims = outputs.shape[1]
header = [f"cddd_{str(i).zfill(3)}" for i in range(num_dims)]

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # header
    for row in outputs:
        writer.writerow(row.tolist())
