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

smiles_list_2 = [
    "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",    # Ibuprofen
    "CC(C)Cc1ccccc1C(=O)O",              # Naproxen
    "CC(C)Cc1cc(cc(c1C)C(=O)O)O",        # Flurbiprofen
    "CC(C)Cc1ccc2c(c1)OCC2",             # Ketoprofen

    "CCOC(=O)c1cccc(c1)C(F)(F)F",        # Ethyl Flufenamate
    "COC(=O)c1ccccc1C(F)(F)F",           # Flufenamic Acid
    "CC(=O)Oc1cccc(c1)C(F)(F)F",         # Mefenamic Acid
    "CCOC(=O)c1ccc(cc1)C(F)(F)F",        # Niflumic Acid

    "CC1=CC(=O)NC(C1)C2=CC=CC=C2",       # Meloxicam
    "CC1=CC(=O)N(C(C1)C2=CC=CC=C2)C3=CC=CC=C3",  # Piroxicam
    "CC1=CC(=O)N(C(C1)C2=CC=CC=C2)C3=CC(=CC=C3)C#N",  # Tenoxicam
    "CC1=CC(=O)N(C(C1)C2=CC=CC=C2)C3=CC(=CC=C3)F",    # Isoxicam

    "CCN1C=NC2=C(N=CN=C12)N",            # Caffeine
    "CCN1C=NC2=C(N=C(N=C2N)N1)N",        # Theophylline
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",      # Theobromine
    "CN1C=NC2=C(N=CN=C12)N(C)C",         # Paraxanthine

    "CC(C)(C)NCC(O)c1ccc(cc1)O",         # Atenolol
    "CC(C)(C)NCC(O)c1ccc(cc1)OC",        # Metoprolol
    "CC(C)(C)NCC(O)c1ccc(cc1)Cl",        # Bisoprolol
    "CC(C)(C)NCC(O)c1ccc(cc1)F",         # Acebutolol
]

def get_toy_dataset():
    """
    Get a list of smiles and a list of targets (logP) for toy model building
    """
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    log_p = [Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0] for x in mols]
    return np.array(smiles_list), np.array(log_p)

def get_drug_dataset():
    """
    Get a list of structurally related drugs to test clustering
    """
    mols = [Chem.MolFromSmiles(x) for x in smiles_list_2]
    log_p = [Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0] for x in mols]
    return np.array(smiles_list), np.array(log_p)

def get_tuned_model():
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)
    model.fit_tune_model()
    return model