import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

class Descriptor:
    def __init__(self, VT, scaler, to_drop):
        self.VT = VT
        self.scaler = scaler
        self.to_drop = to_drop

    def calculate_from_smi(self, smiles):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols])
        fps = self.VT.transform(fps)
        fps = np.delete(fps, self.to_drop, axis=1)
        fps = self.scaler.transform(fps)
        return fps

class GP_qsar:
    def __init__(self, train_smiles, train_y):
        self.train_smiles = train_smiles
        self.y = train_y
        self.X = None
        self.predictor = GaussianProcessRegressor(n_restarts_optimizer=10)
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
        initial_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, 2048) 
                                for smi in self.train_smiles])
        
        train_fps = self.VT.fit_transform(initial_fps)
        corr_matrix = np.corrcoef(train_fps, rowvar=False)
        upper = np.triu(corr_matrix, k=1)
        to_drop = [i for i in range(upper.shape[1]) if any(upper[:, i] > 0.9)]
        train_fps = np.delete(train_fps, to_drop, axis=1)
        train_fps = self.scaler.fit_transform(train_fps)
        self.X = train_fps
        self.descriptor = Descriptor(self.VT, self.scaler, to_drop)

    def fit_tune_model(self):
        self.predictor = tune_hyperparameters(self.predictor, self.X, self.y)
        
    def predict_from_smiles(self, smiles, uncert=False):
        if isinstance(smiles, str):
            smiles = [smiles]
        elif isinstance(smiles, np.ndarray):
            smiles = smiles.tolist()
        
        test_fps = self.descriptor.calculate_from_smi(smiles)
        if uncert:
            predictions, std = self.predictor.predict(test_fps, return_std=True)
            return predictions, std
        else: 
            predictions = self.predictor.predict(test_fps, return_std=False)
            return predictions
        
    def evaluate_acquisition_functions(self, smiles, acquisition_function='EI', y_max = None, kappa=1):

        if y_max == None:
            y_max = np.max(self.y)

        if isinstance(smiles, str):
            smiles = [smiles]
        elif isinstance(smiles, np.ndarray):
            smiles = smiles.tolist()
        
        # Calculate descriptors for input SMILES
        x = self.descriptor.calculate_from_smi(smiles)
        
        # Get GP predictions and uncertainties
        predictions, std = self.predictor.predict(x, return_std=True)

        
        # Evaluate acquisition functions
        if acquisition_function == 'EI':
            # Avoid division by zero by ensuring y_std is positive
            std = np.maximum(std, 1e-9)
            
            # Compute the improvement over the current best
            improvement = predictions - y_max
            
            # Compute the standard normal CDF and PDF
            norm_cdf = norm.cdf(improvement / std)
            norm_pdf = norm.pdf(improvement / std)
            
            # Compute Expected Improvement
            ei = improvement * norm_cdf + std * norm_pdf
            
            return ei
        
        elif acquisition_function == 'PI':
            # Avoid division by zero by ensuring y_std is positive
            std = np.maximum(std, 1e-9)
            
            # Compute the improvement, ensuring it is non-negative
            improvement = np.maximum(predictions - y_max, 0)
            
            # Compute the standard normal CDF for the normalized improvement
            pi = norm.cdf(improvement / std)
            
            return pi
        
        elif acquisition_function == 'UCB':
            # Upper Confidence Bound
            ucb = predictions + kappa * std
            return ucb
        
        else:
            raise ValueError("Unsupported acquisition function. Choose from 'EI', 'PI', or 'UCB'.")


def tune_hyperparameters(gp, X, y):
    kernels = [
        # RBF Kernel
        C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),

        # Matern Kernel with nu=1.5
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=0.5, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=2.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),

        # Matern Kernel with nu=2.5
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=0.5, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=2.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),

        # Combination of RBF and Matern Kernels
        C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),

        # Combination of different kernels with varying parameters
        C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=2.0, length_scale_bounds=(1e-2, 1e2)) * Matern(length_scale=2.0, nu=2.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2)) * Matern(length_scale=0.5, nu=1.5, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(),

        # Simple Kernel Combinations
        C(1.0, (1e-2, 1e2)) * WhiteKernel(),
        C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=1.5),
        C(1.0, (1e-2, 1e2)) * Matern(length_scale=1.0, nu=2.5)
    ]

    param_grid = {
        "alpha": [1e-10, 1e-5, 1e-2, 1e-1, 1e0],
        "kernel": kernels
    }

    grid_search = GridSearchCV(gp, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_#.fit(X, y)