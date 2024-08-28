import pandas as pd
import numpy as np
from rdkit import Chem
from gp_qsar import GP_qsar

smiles_list = [
    "CCO", "CCCC", "CCOCC", "CC(=O)O",
    "CCN", "CC(C)C", "C1CCCCC1", "C1=CC=CC=C1",
    "CC(C)O", "CC(C)C(=O)O", "COC", "CCN(CC)CC",
    "CC(C)(C)O", "C(CN)O", "CNC", "CC(C)C(=O)OCC",
    "CCOCCO", "C1=CC=CC=C1C(=O)O", "CC(C)OCC", "CNC(=O)C"
]

def get_toy_dataset():
    """
    Get a list of smiles and a list of targets (logP) for toy model building
    """
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    log_p = [Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0] for x in mols]
    return np.array(smiles_list), np.array(log_p)

def get_tuned_model():
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)
    # model.fit_tune_model()
    return model

model = get_tuned_model()
print(model.predict_from_smiles("c1ccccc1"))