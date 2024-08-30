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

def test_tuning():
    # test initialisation and tuning of model
    smiles, target = get_toy_dataset()
    model = GP_qsar(smiles, target)
    model.fit_tune_model()

    assert model.X is not None
    assert model.descriptor is not None
    assert model.features[0] == "Physchem"
    assert np.isclose(model.predict_from_smiles(["c1ccccc1OC"])[0],1.717358942834102, atol=0.001)

def test_acquisition_function():
    # test initialisation and tuning of model
    model = get_tuned_model()

    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "UCB")[0],1.8927741567631535, atol=0.001)
    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "EI")[0],8.29969888580552e-06, atol=0.001)
    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "PI")[0],0.00019047019435642292, atol=0.001)