import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
import onnxruntime as ort
import sys


# current file directory
root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(root)
from tokenizer import InputPipelineInferEncode
from preprocessing import preprocess_smiles

root = os.path.dirname(os.path.abspath(__file__))

@dataclass
class HParams:
    """Hyperparameters for the model."""
    batch_size: int = 128

class InferenceModel:
    """CDDD Inference Model for encoding SMILES to embeddings and back."""
    def __init__(self):
        self.hparams = HParams()
        """Initialize the inference model."""
        encoder_path = os.path.join(root, "..", "..", "checkpoints", "encoder.onnx")
        self.encoder_session = ort.InferenceSession(encoder_path)

    def seq_to_emb(
        self, smiles_list: List[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode a list of SMILES strings into molecular descriptors.

        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Optional batch size for inference (default: 128)

        Returns:
            numpy.ndarray: Molecular descriptors for the input SMILES
        """
        if batch_size:
            self.hparams.batch_size = batch_size

        processed_smiles = [preprocess_smiles(smi) for smi in smiles_list]
        good_smiles = []
        accepted_idxs = []
        for i, smiles in enumerate(processed_smiles):
            if str(smiles) == "nan":
                continue
            good_smiles += [smiles]
            accepted_idxs += [i]

        X = np.full((len(smiles_list), 512), np.nan, dtype=np.float32)
        if len(good_smiles) == 0:
            return X

        input_pipeline = InputPipelineInferEncode(good_smiles, self.hparams)
        input_pipeline.initialize()
        emb_list = []
        while True:
            try:
                input_seq, input_len = input_pipeline.get_next()

                outputs = self.encoder_session.run(
                    None,
                    {
                        "Input/Placeholder:0": input_seq.astype(np.int32),
                        "Input/Placeholder_1:0": input_len.astype(np.int32),
                    },
                )
                emb_list.append(outputs[0])
            except StopIteration:
                break
        embeddings = np.vstack(emb_list)
        X[accepted_idxs] = embeddings
        return X
