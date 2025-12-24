import csv
import os
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import sys
import h5py
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

dest_dir = os.path.join(root, "..", "data")

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

input_file = os.path.join(dest_dir, "chembl36_smiles.csv")

MODEL_IDS = [("eos4rw4", np.float16)]

tmp_inputs = os.path.join(dest_dir, "tmp_inputs")
if not os.path.exists(tmp_inputs):
    os.mkdir(tmp_inputs)

tmp_outputs = os.path.join(dest_dir, "tmp_outputs")
if not os.path.exists(tmp_outputs):
    os.mkdir(tmp_outputs)

chunksize = 10000

for i, chunk in tqdm(enumerate(pd.read_csv(input_file, chunksize=chunksize))):
    chunk_file = os.path.join(tmp_inputs, "smiles_{0}.csv".format(str(i).zfill(3)))
    chunk.to_csv(chunk_file, index=False)

file_names = []
for fn in os.listdir(tmp_inputs):
    if fn.startswith("smiles_") and fn.endswith(".csv"):
        file_names.append(fn)
file_names = sorted(file_names)

batch_ids = []

for fn in tqdm(file_names):
    if fn.startswith("smiles_") and fn.endswith(".csv"):
        batch_id = fn.split("_")[1].split(".")[0]
        batch_ids += [int(batch_id)]
        for model_id, _ in MODEL_IDS:
            output_file = os.path.join(tmp_outputs, "{0}_{1}.csv".format(model_id, batch_id))
            if os.path.exists(output_file):
                print("Skipping existing file:", output_file)
                continue
            chunk_input_file = os.path.join(tmp_inputs, fn)
            cmd = "ersilia serve {0}; ersilia -v run -i {1} -o {2} --batch_size 1000; ersilia close".format(model_id, chunk_input_file, output_file)
            subprocess.run(cmd, shell=True, check=True)


batch_ids = sorted(set(batch_ids))

print("Merging data into HDF5 files")
chunksize = 50000
string_dtype = h5py.special_dtype(vlen=str)
for model_id, dtype in tqdm(MODEL_IDS):
    h5_file = os.path.join(dest_dir, "{0}.h5".format(model_id))
    h5_file = os.path.abspath(h5_file)
    smiles_all = []
    print(h5_file)
    with h5py.File(h5_file, "a") as f:
        for i, batch_id in enumerate(batch_ids):
            chunk_file = os.path.join(tmp_outputs, "{0}_{1}.csv".format(model_id, str(batch_id).zfill(3)))
            print(chunk_file)
            if not os.path.exists(chunk_file):
                continue
            df = pd.read_csv(chunk_file)
            features = list(df.columns)[2:]
            before_rows = df.shape[0]
            df = df.dropna(subset=features, how="all")
            after_rows = df.shape[0]
            print("Dropped {0} rows with all NaN features".format(before_rows - after_rows))
            df = df.reset_index(drop=True)
            smiles_list = df["input"].tolist()
            smiles_all += smiles_list
            X = np.array(df[features].values, dtype=dtype)
            if i == 0:
                # Create datasets with variable-length string dtype
                f.create_dataset(
                    "values",
                    data=X,
                    shape=X.shape,
                    maxshape=(None, X.shape[1]),
                    dtype=dtype,
                    chunks=(chunksize, X.shape[1]),
                    compression="gzip",
                )
                f.create_dataset(
                    "input",
                    data=smiles_list,
                    shape=(len(smiles_list),),
                    maxshape=(None,),
                    dtype=string_dtype,
                    compression="gzip",
                )
                f.create_dataset(
                    "features",
                    data=features,
                    shape=(len(features),),
                    maxshape=(None,),
                    dtype=string_dtype,
                    compression="gzip",
                )
            else:
                # Append to datasets
                f["values"].resize((f["values"].shape[0] + X.shape[0]), axis=0)
                f["values"][-X.shape[0]:] = X
                f["input"].resize((f["input"].shape[0] + len(smiles_list)), axis=0)
                f["input"][-len(smiles_list):] = smiles_list

smiles_out = os.path.join(dest_dir, "chembl36_smiles_no_nans.csv")
with open(smiles_out, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smi in smiles_all:
        writer.writerow([smi])

print("Initial smiles:", pd.read_csv(input_file).shape[0])
print("Total Chembl36 smiles post Nan cleaning:", len(smiles_all))
