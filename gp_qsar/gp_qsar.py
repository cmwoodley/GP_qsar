import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import EState
import itertools
from copy import deepcopy
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    WhiteKernel,
    ConstantKernel as C,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

from .utils import Descriptor
from .utils import get_all_descriptors
from .utils import var_corr_scaler

### Define Acquisition functions
# Equations taken from https://github.com/modAL-python/modAL


def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff) / std)


def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)


def UCB(mean, std, kappa):
    return mean + kappa * std

class GP_qsar:
    def __init__(self, train_smiles, train_y):
        self.train_smiles = train_smiles
        self.y = train_y
        self.X = None
        self.predictor = GaussianProcessRegressor(
            n_restarts_optimizer=10, random_state=42
        )
        self.features = []
        self.descriptor = None
        self.metadata = {
            "model_name": "GP_qsar",
            "version": 1.0,
        }

        # Preprocessing tools
        self.VT = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()

        # Fit the model with training data
        self._prepare_data()

    def _prepare_data(self):
        self.feature_dict = get_all_descriptors(self.train_smiles)

    def fit_tune_model(self):
        (
            self.predictor,
            self.X,
            self.features,
            self.VT,
            self.scaler,
            to_drop,
        ) = tune_model(self.predictor, self.feature_dict, self.y, self.VT, self.scaler)
        self.descriptor = Descriptor(self.VT, self.scaler, to_drop, self.features)

    def predict_from_smiles(self, smiles, uncert=False):
        if isinstance(smiles, str):
            smiles = [smiles]

        test_fps = self.descriptor.calculate_from_smi(smiles)
        if uncert:
            predictions, std = self.predictor.predict(test_fps, return_std=True)
            return predictions, std
        else:
            predictions = self.predictor.predict(test_fps, return_std=False)
            return predictions

    def evaluate_acquisition_functions(
        self, smiles, acquisition_function="EI", y_max=None, kappa=1.0, tradeoff=0.0
    ):
        if y_max == None:
            y_max = np.max(self.y)

        if isinstance(smiles, str):
            smiles = [smiles]

        # Calculate descriptors for input SMILES
        x = self.descriptor.calculate_from_smi(smiles)

        # Get GP predictions and uncertainties
        predictions, std = self.predictor.predict(x, return_std=True)

        # Evaluate acquisition functions
        if acquisition_function == "EI":
            ei = EI(predictions, std, y_max, tradeoff)
            return ei

        elif acquisition_function == "PI":
            pi = PI(predictions, std, y_max, tradeoff)
            return pi

        elif acquisition_function == "UCB":
            ucb = UCB(predictions, std, kappa)
            return ucb

        else:
            raise ValueError(
                "Unsupported acquisition function. Choose from 'EI', 'PI', or 'UCB'."
            )


def tune_model(gp, feature_dict, y, VT, scaler):
    feature_combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations(list(feature_dict.keys()), r)
            for r in range(1, len(feature_dict.keys()) + 1)
        )
    )
    best_models = []
    best_score = []

    for features in feature_combinations:
        gp_temp = deepcopy(gp)
        VT_temp = deepcopy(VT)
        scaler_temp = deepcopy(scaler)

        X = np.concatenate([feature_dict[feat] for feat in features], axis=1)
        X, VT_temp, scaler_temp, _ = var_corr_scaler(X, VT_temp, scaler_temp)

        gp_temp, score = tune_hyperparameters(gp_temp, X, y)
        best_models.append(gp_temp)
        best_score.append(score)

    best_features = feature_combinations[np.argmax(best_score)]
    X = np.concatenate([feature_dict[feat] for feat in best_features], axis=1)
    X, VT, scaler, to_drop = var_corr_scaler(X, VT, scaler)

    gp = best_models[np.argmax(best_score)]

    return gp, X, best_features, VT, scaler, to_drop


def tune_hyperparameters(gp, X, y):
    kernels = [
        # RBF Kernel with more conservative bounds
        C(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 10.0)) + WhiteKernel(),
        C(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(),
        # Matern Kernel with nu=1.5, more conservative bounds
        C(1.0, (0.1, 10.0))
        * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(),
        C(1.0, (0.1, 10.0))
        * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(),
        # Simpler Kernel Combinations with conservative bounds
        C(1.0, (0.1, 10.0)) * WhiteKernel(),
        C(1.0, (0.1, 10.0)) * RBF(1.0, (0.1, 10.0)) * WhiteKernel(),
        C(1.0, (0.1, 10.0)) * Matern(length_scale=1.0, nu=1.5) * WhiteKernel(),
    ]

    param_grid = {
        "alpha": [1e-2, 1e-1, 1e0, 1e1],  # More regularization to avoid overfitting
        "kernel": kernels,
    }

    grid_search = GridSearchCV(
        gp, param_grid=param_grid, cv=10, n_jobs=-1, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_score_
