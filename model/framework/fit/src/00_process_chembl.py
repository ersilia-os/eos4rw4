import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from standardiser import standardise

def standardise_smiles(df, smi_col):
    std_smiles = []
    for smiles in tqdm(list(df[smi_col])):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = standardise.run(mol)
            std_smiles.append(Chem.MolToSmiles(mol))
        except:
            std_smiles.append(None)
    df["std_smiles"] = std_smiles
    df = df[df["std_smiles"].notnull()]
    return df

df = pd.read_csv("../data/chembl_36_chemreps.txt", sep="\t", dtype=str)
print(df.shape)
df = df.dropna()
df = df.drop_duplicates(subset= ["standard_inchi_key"], keep="first")
print(df.shape)
df = standardise_smiles(df, "canonical_smiles")
df = df[~df["std_smiles"].isna()]
print(df.shape)
df = df[["std_smiles"]]
df = df.drop_duplicates(subset=["std_smiles"], keep="first")
print(df.shape)
df.reset_index(drop=True, inplace=True)
df.to_csv("../data/chembl36_smiles.csv", index=False)