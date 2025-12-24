import os
from rdkit import __version__ as rdkit_version
from FPSim2 import FPSim2Engine

root = os.path.dirname(os.path.abspath(__file__))

print(rdkit_version)
assert rdkit_version == "2025.09.1", "Please use RDKit 2025.09.1"

class ChemblNearestNeighbour(object):

    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.fp_database = os.path.join(root, "..", "..", "checkpoints", "fpsim2_database_chembl.h5")
        self.fpe = FPSim2Engine(self.fp_database)
    
    def highest_similarity(self, smiles, metric="tanimoto"):
        results = self.fpe.top_k(
            smiles,
            k=1,
            threshold=0.0,
            metric=metric,
            n_workers=1
        )
        if results is None or len(results) == 0:
            return None, None
        print(results)
        idx = results[0][0]
        print(idx)
        return idx

