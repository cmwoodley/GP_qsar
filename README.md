# GP_qsar

Wrapper around Sklearn gaussian process model to enable use in Reinvent as a scoring function and for use in active learning. Use in REINVENT4 is identical to serialised QSARtuna models.

Features:
* Automates feature selection and hyperparameter tuning/ kernel selection. 
* Produces predictions from SMILES
* Evaluates acquisiton functions for use in active learning

Example config .toml files for use of these models in REINVENT4 are given in /example_config.

# Installation
* Clone this repository
```
git clone https://github.com/cmwoodley/GP_qsar.git
```

* Install GP_qsar
```
pip install .
```

# Example Usage
```
from gp_qsar import GP_qsar
import numpy as np

# Toy dataset for simple example

smiles = np.array([
    "CCO", "C1CCCCC1", "O=C=O", "CC(C)C",
    "C1=CC=CC=C1", "CCN(CC)CC", "C1=CC(=O)NC(=O)N1", "CC(C)O",
    "C#N", "C=O", "O=C(O)C", "CC(C)CC",
    "NCCO", "CC(=O)O", "C1CC1", "O=S(=O)(O)O",
    "CNC", "C=CC", "CCOCC", "CCOC"
])

test_smiles = [
    "C1CCOC1",  # Tetrahydrofuran (THF)
    "N#CCN",    # Cyanogen
    "CC(C)CO",  # Isobutanol
    "C1=CC(=O)OC=C1",  # Furan-2(5H)-one
    "C=C",      # Ethene (Ethylene)
]

y = [
    3.14, 2.718, 1.618, 0.577,
    6.022, 9.81, 1.414, 2.302,
    0.693, 4.669, 0.007, 299792.458,
    1.732, 42.0, 0.001, 8.314,
    1.96, 0.333, 0.618, 1.12
]

# Initialise model
model = GP_qsar(smiles, y)

# Generate predictions 
predictions = model.predict_from_smiles(test_smiles)
predictions_std = model.predict_from_smiles(test_smiles, uncert=True) # Generate with uncertainty

# Evaluate acquisition function
UCB = model.evaluate_acquisition_functions(test_smiles, "UCB")
```

# To do
* Add teach functionality to re-train models with newly acquired datat
* Actually implement metadata to show model performance
* Store names of selected features in some meaningful way
* Improve testing framework
