import pytest
import numpy as np
from gp_qsar.gp_qsar import Descriptor, GP_qsar, tune_hyperparameters
from .store import get_toy_dataset
from .store import get_tuned_model

def test_gp_qsar_initialization():
    # Test the initialization of GP_qsar class
    smiles = ['CCO', 'CCN', 'CCC', 'COC',"CCOC","c1ccccc1"]
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gp_model = GP_qsar(train_smiles=smiles, train_y=y)
    
    assert gp_model.train_smiles == smiles
    assert np.array_equal(gp_model.y, y)
    assert gp_model.X is not None
    assert gp_model.descriptor is not None

def test_tuning():
    # test initialisation and tuning of model
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)
    model.fit_tune_model()

    assert np.isclose(model.predict_from_smiles(["c1ccccc1OC"])[0],0.5233897, atol=0.001)

def test_without_tuning():
    # test whether functions run without training model
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)  

    assert float(model.predict_from_smiles("c1ccccc1")[0]) == 0.
