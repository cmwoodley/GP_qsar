import pandas as pd
import numpy as np
from rdkit import Chem
from gp_qsar import GP_qsar, GPytorch_qsar

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

# LogD data taken from Wang et al. Journal of Chemometrics, 29(7), 389-398.

censored_smiles = [
    'OC[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)CNC',
    'O1c2c(ccc(O[C@@H]3O[C@H](C(O)=O)[C@@H](O)[C@H](O)[C@H]3O)c2)C(=CC1=O)C',
    'S1C[C@H](O[C@H]1CO)N1C=CC(=NC1=O)N',
    'S(=O)(=O)(Nc1onc(C)c1C)c1ccc(N)cc1',
    'OC=1C(=O)C=CN(CC)C=1C',
    'OC(=O)C(C)c1cc(ccc1)\\C(=N/O)\\c1ccccc1',
    'Fc1cc(F)ccc1[NH+]1c2nc(N3C[C@@H]4[C@H](C3)C4N)c(F)cc2C([O-])=C(C(O)=O)[CH-]1',
    'OC=1C(=O)C=CN(CC=C)C=1CC',
    '[Na+].Fc1ccc(cc1)-c1n(\\C=C\\[C@@H](O)C[C@@H](O)CC(=O)[O-])c(nc1-c1ccc(F)cc1)C(F)(F)F',
    'OCCN(CCO)c1nnc(Nn2c(ccc2C)C)cc1',
    'O=C1N(c2nc(nc(c2CC1(C)C)C)C)Cc1ccc(cc1)-c1ccccc1-c1nn[nH]n1',
    'O=C1N(c2nc(nc(c2CC1)C(C)C)C)Cc1ccc(cc1)-c1ccccc1-c1nn[nH]n1',
    'O=C1N(c2c(cccc2)C(=N[C@H]1NC(=O)Nc1cc(N(C)c2nn[nH]n2)c(cc1)C)c1ccccc1)C',
    'O1c2c(OCC1c1[nH]ccn1)cccc2',
    'n1c2c(n(c1)Cc1cc(N)ccc1)cccc2',
    'O=C1C=C2[C@@]([C@H]3[C@@H]([C@H]4CC[C@](O)(C(=O)CO)[C@@]4(C[C@H]3O)C)C[C@H]2C)(C=C1)C',
    'Oc1cc(ccc1)[C@H](N1C[C@H](N(C[C@@H]1C)Cc1ccccc1)C)c1ccc(cc1)-c1ccccc1C(O)=O',
    'FC(F)(F)c1ccc(cc1)/C(=N/OCCN)/CCCCOC',
    's1c2c(nc1O)c(O)ccc2CCNCCNS(=O)(=O)CCOCCc1cc(cc(c1)C)C',
    's1c2c(nc1O)c(O)ccc2CCNCCCCCCOCCc1ccccc1',
    'OC=1C(=O)C=CN(CCCCCCCC)C=1C',
    'S(CC)C=1OC(=O)c2c(N=1)cccc2',
    's1c2c(nc1O)c(O)ccc2CCNCCC(F)(F)CCCOCCc1ccccc1',
    'Clc1cc2c(Oc3c(N=C2N2CCNCC2)cccc3)cc1',
    'Fc1ccc(cc1C(F)(F)F)COC[C@@H]1N(C(=O)CNC1)c1ccc(OCCCOCc2ccccc2OC)cc1',
    'O(c1ccc(cc1OC)C)c1ncccc1/C(=N/O)/NC1CCCCC1',
    'Clc1cc(Cl)ccc1[C@@]1(O[C@H](CO1)COc1ccc(N2CCN(CC2)C(=O)C)cc1)Cn1ccnc1',
    'Fc1cc(ccc1)-c1nn2c(N=C(C=C2NCCCN2CCCC2)c2ccccc2)c1',
    'Ic1cc(cc(I)c1OCCN1CCCC1)C(=O)c1c2c(oc1CCCC)cccc2'
]

LogD = np.array([1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.06, 1.21,
       1.41, 1.58, 1.76, 1.92, 2.1 , 2.3 , 2.41, 2.59, 2.73, 2.88, 3.  ,
       3.26, 3.48, 3.7 , 4.  , 4.  , 4.  , 4.  ])

censored = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1])

def get_toy_dataset():
    """    Get a list of smiles and a list of targets (logP) for toy model building
    """
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    log_p = [Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0] for x in mols]
    return np.array(smiles_list), np.array(log_p)

def get_drug_dataset():
    """    Get a list of structurally related drugs to test clustering
    """
    mols = [Chem.MolFromSmiles(x) for x in smiles_list_2]
    log_p = [Chem.rdMolDescriptors.CalcCrippenDescriptors(x)[0] for x in mols]
    return np.array(smiles_list), np.array(log_p)

def get_tuned_model():
    """    Get a tuned sklearn gaussian process model
    """
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)
    model.fit_tune_model()
    return model

def get_tuned_gpytorch_model():
    """    Get a tuned Gpytorch gaussian process model
    """
    smiles, target = get_toy_dataset()
    model = GPytorch_qsar(smiles, target)
    model.fit_tune_model(n_trials=20)
    return model

def get_censored_dataset():
    """ Get a dataset with censored data to test implementation of custom survival loss
    """
    return censored_smiles, LogD, censored