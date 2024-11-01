import pytest
import numpy as np
from gp_qsar.gpytorch_qsar import GPytorch_qsar
from .store import get_censored_dataset
from .store import get_drug_dataset
from .store import get_toy_dataset
from .store import get_tuned_gpytorch_model

def test_gpytorch_qsar_initialization():
    # Test the initialization of GP_qsar class
    smiles = ['CCO', 'CCN', 'CCC', 'COC',"CCOC","c1ccccc1"]
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gp_model = GPytorch_qsar(train_smiles=smiles, train_y=y)
    
    assert gp_model.X is None
    assert gp_model.descriptor is None    
    assert gp_model.train_smiles == smiles
    assert np.array_equal(gp_model.y, y)

def test_gpytorch_tuning():
    # test initialisation and tuning of model
    smiles, target = get_toy_dataset()
    model = GPytorch_qsar(smiles, target)
    model.fit_tune_model()

    assert model.X is not None
    assert model.censored is None
    assert model.descriptor is not None
    assert model.features[0] == "FCFP"

def test_gyptorch_censored_data():
    # test initialisation and tuning of model
    censored_smiles, LogD, censored = get_censored_dataset()
    model = GPytorch_qsar(censored_smiles, LogD, censored)
    model.fit_tune_model()

    assert model.X is not None
    assert model.censored is not None
    assert model.descriptor is not None
    assert model.features[0] == "ECFP"

def test_gpytorch_predict_from_smiles():
    model = get_tuned_gpytorch_model()

    # Predict from list
    assert np.isclose(model.predict_from_smiles(["c1ccccc1OC", "c1ccccc1OCC"])[0],0.798472585796927, atol=0.001)

    # Predict from np.ndarray
    assert np.isclose(model.predict_from_smiles(np.array(["c1ccccc1OC", "c1ccccc1OCC"]))[0],0.798472585796927, atol=0.001)

    # test predict with uncertainty
    assert np.isclose(model.predict_from_smiles(["c1ccccc1OC", "c1ccccc1OCC"], uncert=True)[1][1], 0.9215212070184364, atol=0.001)

    # Predict from string
    assert np.isclose(model.predict_from_smiles("c1ccccc1OC")[0],0.798472585796927, atol=0.001)

def test_gpytorch_acquisition_function():
    # test initialisation and tuning of model
    model = get_tuned_gpytorch_model()
    
    # AF from list
    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "UCB")[0],1.719357114040894, atol=0.001)
    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "EI")[0],0.01791102687453268, atol=0.001)
    assert np.isclose(model.evaluate_acquisition_functions(["c1ccccc1OC"], "PI")[0],0.0470048577133137, atol=0.001)

    # AF from np.ndarray
    assert np.isclose(model.evaluate_acquisition_functions(np.array(["c1ccccc1OC"]), "PI")[0],0.0470048577133137, atol=0.001)

    # AF from string
    assert np.isclose(model.evaluate_acquisition_functions("c1ccccc1OC", "UCB")[0],1.719357114040894, atol=0.001)

    # Unsupported function
    with pytest.raises(ValueError):
        model.evaluate_acquisition_functions("c1ccccc1OC", "not a function")

@pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.GPInputWarning")
def test_gpytorch_batch_selection():
    model = get_tuned_gpytorch_model()

    candidates = get_drug_dataset()[0]

    # Test greedy batch selection
    batch = model.batch_selection(candidates, "greedy", 4)
    assert batch == [12, 0, 1, 2]

    # Test gpo batch selection
    batch, probs = model.batch_selection(candidates, "gpo", 4)

    assert list(batch) == [6, 1, 5, 7]
    assert list(probs) == [0.1952, 0.1239, 0.1193, 0.1153]

    # Test parallel Thompson sampling
    batch = model.batch_selection(candidates, "ts", 4)
    assert list(batch) == [1, 5, 15, 16]

    # Test seed and batch size
    batch, probs = model.batch_selection(candidates, "gpo", 4, seed=52)
    assert list(probs) != [0.1952, 0.1239, 0.1193, 0.1153]
    batch = model.batch_selection(candidates, "ts", 8, seed=52)
    assert len(batch) == 8

    # Test invalid method
    with pytest.raises(ValueError):
        model.batch_selection(candidates, "not a function",4)