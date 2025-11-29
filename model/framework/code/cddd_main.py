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

    def seq_to_emb(self, smiles_list: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode a list of SMILES strings into molecular descriptors.

        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Optional batch size for inference (default: 128)

        Returns:
            numpy.ndarray: Molecular descriptors for the input SMILES
        """
        if batch_size:
            self.hparams.batch_size = batch_size

        # Process SMILES and get preprocessed versions
        processed_smiles = [preprocess_smiles(smi) for smi in smiles_list]
        valid_mask = [not pd.isna(smi) for smi in processed_smiles]
        valid_smiles = [smi for smi, valid in zip(processed_smiles, valid_mask) if valid]
        
        if not valid_smiles:
            raise ValueError("No valid SMILES found after preprocessing")
            
        # Initialize input pipeline with valid SMILES
        input_pipeline = InputPipelineInferEncode(valid_smiles, self.hparams)
        input_pipeline.initialize()
        emb_list = []
        
        while True:
            try:
                # Get next batch
                input_seq, input_len = input_pipeline.get_next()
                
                # Run inference using ONNX Runtime
                outputs = self.encoder_session.run(
                    None,  # output names - passing None means return all outputs
                    {
                        'Input/Placeholder:0': input_seq.astype(np.int32),
                        'Input/Placeholder_1:0': input_len.astype(np.int32)
                    }
                )
                emb_list.append(outputs[0])
            except StopIteration:
                break
            
        if emb_list:
            embeddings = np.vstack(emb_list)
        else:
            embeddings = np.array([])
            
        # Create a mapping of original SMILES to their embeddings
        result_embeddings = []
        for smi, valid in zip(smiles_list, valid_mask):
            if valid:
                result_embeddings.append(embeddings[0])
                embeddings = embeddings[1:]
            else:
                result_embeddings.append(np.full(embeddings[0].shape, np.nan))
                
        return np.array(result_embeddings)