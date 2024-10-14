import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import EState
import itertools
from copy import deepcopy
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
import torch
import gpytorch

from .utils import Descriptor
from .utils import get_all_descriptors
from .utils import var_corr_scaler

### Define Acquisition functions

def custom_log_likelihood(model_output: gpytorch.distributions.MultivariateNormal, target: torch.Tensor, censoring_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate the Censored Mean Squared Error (Censored MSE) loss.
    
    Args:
        predictions (torch.Tensor): The model's predictions (μn).
        true_labels (torch.Tensor): The true labels (yn).
        censoring_mask (torch.Tensor): Censorship flags (mn). 
                                     -1 for left-censored, 0 for uncensored, 1 for right-censored.
    
    Returns:
        torch.Tensor: The Censored MSE loss.
    """
    # Get the mean and variance from the likelihood output
    mean = model_output.mean
    variance = model_output.variance
    stddev = torch.sqrt(variance)

    # Use regular mll if no censored mask provided
    if censoring_mask is None:
        censoring_mask = np.zeros(target.shape[0])


    # Initialize log_likelihood to zero
    log_likelihood = torch.zeros_like(target)

    # Mask for uncensored data (censoring_mask == 0)
    uncensored_mask = (censoring_mask == 0)

    if uncensored_mask.any():
        uncensored_targets = target[uncensored_mask]
        uncensored_means = mean[uncensored_mask]
        uncensored_stddevs = stddev[uncensored_mask]
        uncensored_log_probs = (-0.5 * torch.log(2 * torch.pi * variance[uncensored_mask])
                                - 0.5 * ((uncensored_targets - uncensored_means) ** 2) / variance[uncensored_mask])
        log_likelihood[uncensored_mask] = uncensored_log_probs

    # Mask for right-censored data (censoring_mask == 1)
    right_censored_mask = (censoring_mask == 1)

    if right_censored_mask.any():
        survival_probs = 1 - torch.distributions.Normal(mean[right_censored_mask], stddev[right_censored_mask]).cdf(target[right_censored_mask])
        log_likelihood[right_censored_mask] = torch.log(survival_probs)

    # Mask for left- data (censoring_mask == -1)
    left_censored_mask = (censoring_mask == -1)

    if left_censored_mask.any():
        cdf_probs = torch.distributions.Normal(mean[left_censored_mask], stddev[left_censored_mask]).cdf(target[left_censored_mask])
        log_likelihood[left_censored_mask] = torch.log(cdf_probs)

    return log_likelihood.mean()

def batch_tanimoto_sim(x1: torch.Tensor, x2: torch.Tensor):
    """tanimoto between two batched tensors, across last 2 dimensions"""
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    return (dot_prod) / (x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod)


class TanimotoKernel(gpytorch.kernels.Kernel):
    """Tanimoto coefficient kernel"""

    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)
    
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = None

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# Equations taken from https://github.com/modAL-python/modAL

def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff) / std)


def EI(mean, std, max_val, tradeoff):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)


def UCB(mean, std, kappa):
    return mean + kappa * std

class GPytorch_qsar:
    def __init__(self, train_smiles, train_y, censored=None):
        self.train_smiles = train_smiles
        self.y = np.ravel(train_y)
        self.censored = censored
        self.X = None
        self.predictor = None
        self.likelihood = None
        self.features = []
        self.descriptor = None
        self.metadata = {
            "model_name": "GPytorch_qsar",
            "version": 1.0,
        }

        # Preprocessing tools
        self.VT = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # Fit the model with training data
        self._prepare_data()

    def _prepare_data(self):
        self.feature_dict = get_all_descriptors(self.train_smiles)

    def fit_tune_model(self, n_splits = 5):
            
        y_scaled = self.y_scaler.fit_transform(self.y.reshape(-1, 1)).ravel()

        (
            params,
            self.X,
            self.features,
            self.VT,
            self.scaler,
            to_drop,
        ) = tune_model(self.feature_dict, y_scaled, self.VT, self.scaler, self.censored, n_splits)
        self.descriptor = Descriptor(self.VT, self.scaler, to_drop, self.features)

        # Define kernel based on the optimised params
        if params["kernel"] == 'Matern':
            lengthscale=params["lengthscale"]
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(lengthscale=lengthscale,
                                            lengthscale_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
            )
        elif params["kernel"] == 'RBF':
            lengthscale = params["lengthscale"]
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale=lengthscale,
                                        lengthscale_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
            )
        elif params["kernel"] == 'Periodic':
            period_length = params['period_length']
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel(period_length=period_length,
                                                period_length_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
            )
        elif params["kernel"] == 'Tanimoto':
            kernel = gpytorch.kernels.ScaleKernel(TanimotoKernel())  

        noise_level = params["noise"]
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_level)
    )      

        train_X = torch.tensor(self.X)
        train_y = torch.tensor(y_scaled)
        if self.censored is None:
            train_censored = None
        else:
            train_censored = torch.tensor(self.censored)

        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)
            
        self.predictor = GPModel(train_X, train_y, likelihood=self.likelihood)

        self.predictor.train()
        self.likelihood.train()            

        lr = params["lr"]
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        n_iter = params["n_iter"]

        for i in range(n_iter):
            optimizer.zero_grad()
            output = self.predictor(train_X)
            loss = -custom_log_likelihood(self.likelihood(output), train_y, train_censored)
            loss.backward()
            optimizer.step()

    def predict_from_smiles(self, smiles, uncert=False):
        self.predictor.eval()
        self.likelihood.eval()

        if isinstance(smiles, str):
            smiles = [smiles]

        test_fps = torch.tensor(self.descriptor.calculate_from_smi(smiles))
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_output = self.predictor(test_fps)
            test_output = self.likelihood(test_output)
            predictions = test_output.mean.detach().numpy()
            # inverse scale predictions
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1,1))
        if uncert:
            std = test_output.stdev.numpy()
            return predictions, std
        else:
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


def tune_model(feature_dict, y, VT, scaler, censored, n_splits = 5):

    VT_temp = deepcopy(VT)
    scaler_temp = deepcopy(scaler)

    model_params, score = tune_hyperparameters(feature_dict, VT_temp, scaler_temp, y, censored, n_splits)

    best_features = model_params["features"].split(",")

    X = np.concatenate([feature_dict[feat] for feat in best_features], axis=1)
    X, VT, scaler, to_drop = var_corr_scaler(X, VT, scaler)

    return model_params, X, best_features, VT, scaler, to_drop

import optuna

def objective(trial, feature_dict, VT, scaler, y, censored, n_splits):

    feature_combinations = [
        ','.join(features) for features in itertools.chain.from_iterable(
            itertools.combinations(list(feature_dict.keys()), r)
            for r in range(1, len(feature_dict.keys()) + 1)
        )
    ]

    feature_combinations = [x for x in feature_combinations if x != "Physchem"]



    # Use Optuna's suggest_categorical with the string-encoded feature combinations
    features = trial.suggest_categorical("features", feature_combinations)

    # Convert the selected string back to a tuple (if necessary)
    features = features.split(',')
    X = np.concatenate([feature_dict[feat] for feat in features], axis=1)
    X, VT, scaler, _ = var_corr_scaler(X, VT, scaler)
    
    # Suggest different kernel types
    kernel_type = trial.suggest_categorical('kernel', ['Matern', 'RBF', 'Periodic', 'Tanimoto'])

    # Define kernel based on the suggestion
    if kernel_type == 'Matern':
        lengthscale = trial.suggest_float('lengthscale', 0.1, 10.0)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(lengthscale=lengthscale,
                                          lengthscale_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
        )
    elif kernel_type == 'RBF':
        lengthscale = trial.suggest_float('lengthscale', 0.1, 10.0)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=lengthscale,
                                       lengthscale_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
        )
    elif kernel_type == 'Periodic':
        period_length = trial.suggest_float('period_length', 0.1, 5.0)
        kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(period_length=period_length,
                                            period_length_prior=gpytorch.priors.NormalPrior(1.0, 0.5))
        )
    elif kernel_type == 'Tanimoto':
        kernel = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    # Suggest likelihood noise level
    noise_level = trial.suggest_float('noise', 1e-4, 1e-2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_level)
    )

    # Cross-validation setup
    kf = KFold(n_splits=n_splits)
    scores = []


    n_iter = trial.suggest_int('n_iter', 10, 200)
    lr=trial.suggest_float('lr', 1e-5, 1e-1)

    for train_index, test_index in kf.split(X):
        train_x = torch.tensor(X[train_index], dtype=torch.float32)
        test_x = torch.tensor(X[test_index], dtype=torch.float32)
        train_y = torch.tensor(y[train_index], dtype=torch.float32)
        test_y = torch.tensor(y[test_index], dtype=torch.float32)

        if censored is None:
            censored_train, censored_test = None, None
        else:
            censored_train = torch.tensor(censored[train_index], dtype=torch.float32)
            censored_test = torch.tensor(censored[test_index], dtype=torch.float32)


        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)
            
        # Initialize model and likelihood
        model = GPModel(train_x, train_y, likelihood)

        # Training loop
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -custom_log_likelihood(likelihood(output), train_y, censored_train)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output_test = model(test_x)
            val_loss = -custom_log_likelihood(likelihood(output_test), test_y, censored_test)
            scores.append(val_loss.item())

    # Return the average cross-validation loss
    return np.mean(scores)


def tune_hyperparameters(feature_dict, VT, scaler, y, censored=None, n_splits=5, n_trials=50):
    study = optuna.create_study(direction='minimize')

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, feature_dict, VT, scaler, y, censored, n_splits), n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best score: {study.best_value}")
    return study.best_params, study.best_value