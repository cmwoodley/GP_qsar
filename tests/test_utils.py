import pytest
# import sys
# sys.path.insert(1,"./")
from .store import get_drug_dataset
from .store import get_toy_dataset
from gp_qsar.utils import splitter
import numpy as np

smiles_list = [
    "CCO", "CCCC", "CCOCC", "CC(=O)O",
    "CCN", "CC(C)C", "C1CCCCC1", "C1=CC=CC=C1",
    "CC(C)O", "CC(C)C(=O)O", "COC", "CCN(CC)CC",
    "CC(C)(C)O", "C(CN)O", "CNC", "CC(C)C(=O)OCC",
    "CCOCCO", "C1=CC=CC=C1C(=O)O", "CC(C)OCC", "CNC(=O)C"
]
def test_splitting():
    smiles, y = get_toy_dataset()

    smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y)
    smiles_tr_want = ['CC(C)O','CC(C)C','CCN(CC)CC','CC(=O)O','CC(C)OCC','CCOCCO','C(CN)O',
                    'CCOCC','CC(C)C(=O)O','CNC(=O)C','CCN','CC(C)(C)O','C1=CC=CC=C1','COC',
                    'CNC','C1CCCCC1']
    y_te_want = np.array([-1.4000e-03,1.3848e+00,1.2055e+00,1.8064e+00])

    assert all(smiles_tr[i] == smiles_tr_want[i] for i in range(len(smiles_tr)))
    assert np.allclose(np.array(y_te), y_te_want)

    smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, "stratified", n_bins=2)
    smiles_te_want = ['CC(C)O','CC(C)C','CCN(CC)CC','CNC']
    y_tr_want = [1.8064e+00,2.3406e+00,1.0428e+00,-1.4000e-03,7.7720e-01,2.6260e-01,
                -1.0626e+00,1.4313e+00,7.2700e-01,-3.5000e-02,1.3848e+00,1.6866e+00,
                1.5200e-02,-2.4770e-01,1.2055e+00,9.0900e-02]

    assert all(smiles_te[i] == smiles_te_want[i] for i in range(len(smiles_te)))
    assert np.allclose(np.array(y_tr), y_tr_want)

    # Test None n_bins
    with pytest.raises(ValueError):
        smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, method="stratified",n_bins=None)

    # Test invalid splitting method
    with pytest.raises(ValueError):
        smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, method="not_a_method")

def test_structure_splitting():
    smiles, y = get_drug_dataset()

    # Test all singleton clusters error
    with pytest.raises(ValueError):
        smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, method="scaffold",cutoff=0.3)

    # Test None cutoff
    with pytest.raises(ValueError):
        smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, method="scaffold",cutoff=None)

    # Test split
    smiles_tr, smiles_te, y_tr, y_te = splitter(smiles, y, method="scaffold",cutoff=0.7)
    smiles_tr_want = ['CC(C)C','C1CCCCC1','C1=CC=CC=C1','COC','CC(C)(C)O','C(CN)O','CNC',
                    'CCOCCO','C1=CC=CC=C1C(=O)O','CC(C)OCC','CNC(=O)C','CCO','CCN','CC(=O)O',
                    'CC(C)C(=O)O']
    y_tr_want = np.array([2.492,2.6307,2.8821,3.98258,0.4284,-0.0991,-1.0293,1.8137,
                2.1167,2.7615,2.2472,3.0732,2.8821,2.82,4.1109])
    assert all(smiles_tr[i] == smiles_tr_want[i] for i in range(len(smiles_tr)))
    assert np.allclose(np.array(y_tr), y_tr_want)