import os
import pandas as pd
from ftplib import FTP
import csv
from tqdm import tqdm
from FPSim2.io import create_db_file
from rdkit import Chem
from rdkit import __version__ as rdkit_version

print(rdkit_version)
assert rdkit_version == "2025.09.1", "Please use RDKit 2025.09.1"

root = os.path.dirname(os.path.abspath(__file__))

dest_dir = os.path.join(root, "..", "data")

print("Indexing ChEMBL...")
with open(os.path.join(root, "..", "data", "chembl36_smiles_no_nans.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = []
    for r in reader:
        smiles_list += [r[0]]
print(smiles_list[:10])

mols = [[smiles, i] for i, smiles in enumerate(smiles_list)]

print("Creating a database file with Morgan fingerprints")

create_db_file(
    mols_source=mols,
    filename=os.path.join(dest_dir, "fpsim2_database_chembl.h5"),
    mol_format='smiles',
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 1024}
)

with open(os.path.join(dest_dir, "fpsim2_database_chembl_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "index"])
    for smiles, i in mols:
        writer.writerow([smiles, i])


print("Done creating the database file!")
