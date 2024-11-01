# Importing modules
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import EState
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class Descriptor:
    def __init__(self, VT, scaler, to_drop, features):
        self.VT = VT
        self.scaler = scaler
        self.to_drop = to_drop
        self.features = features

    def calculate_from_smi(self, smiles):
        feature_dict = get_all_descriptors(smiles)
        fps = np.concatenate([feature_dict[feat] for feat in self.features], axis=1)
        fps = self.VT.transform(fps)
        fps = np.delete(fps, self.to_drop, axis=1)
        fps = self.scaler.transform(fps)
        return fps


def get_all_descriptors(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    descriptors = {}
    descriptors["ECFP"] = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
    )
    descriptors["FCFP"] = np.array(
        [
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True)
            for mol in mols
        ]
    )
    descriptors["Physchem"] = np.array(
        [
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.Chi0n(mol),
                Descriptors.Chi1n(mol),
                Descriptors.Chi2n(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.Kappa3(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.HallKierAlpha(mol),
                EState.EState_VSA.EState_VSA1(mol),  # EState_VSA bin 1
            ]
            for mol in mols
        ]
    )
    return descriptors


def var_corr_scaler(X, VT, scaler):
    X = VT.fit_transform(X)
    corr_matrix = np.corrcoef(X, rowvar=False)
    upper = np.triu(corr_matrix, k=1)
    to_drop = [i for i in range(upper.shape[1]) if any(upper[:, i] > 0.9)]
    X = np.delete(X, to_drop, axis=1)
    X = scaler.fit_transform(X)
    return X, VT, scaler, to_drop


def stratified_split(smiles, y, random_state, test_size, n_bins):
    """
    Stratified split by binning target variable and random sampling from each bin.
    """
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    y = np.array(y).reshape(-1, 1)
    y_binned = binner.fit_transform(y).astype(int).ravel()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_index, test_index = next(splitter.split(y, y_binned))

    smiles_tr, smiles_te = np.array(smiles)[train_index], np.array(smiles)[test_index]
    y_tr, y_te = y[train_index], y[test_index]

    return smiles_tr, smiles_te, np.ravel(y_tr), np.ravel(y_te)


def butina_cluster(mol_list, cutoff=0.3):
    """
    Butina clustering algorithm, taken from https://github.com/PatWalters/workshop/blob/master/clustering/taylor_butina.ipynb
    """
    fp_list = [rdmd.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_id_list[member] = idx
    return cluster_id_list, fp_list


def scaffold_split(smiles, y, random_state, test_size, cutoff):
    """
    Scaffold split by Butina clustering.
    """
    mol_list = [Chem.MolFromSmiles(x) for x in smiles]
    cluster_id_list, fp_list = butina_cluster(mol_list, cutoff)

    cluster_dict = {}
    for i, cluster in enumerate(cluster_id_list):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = [i]
        else:
            cluster_dict[cluster].append(i)

    train_index, test_index = [], []

    singleton_clusters = [
        cluster for cluster, idx_list in cluster_dict.items() if len(idx_list) == 1
    ]
    # Raise error if all singleton clusters - to do: default to clustering with set number of clusters
    if len(singleton_clusters) == len(smiles):
        raise ValueError(
            "Clustering produced all singleton clusters. Try increasing cutoff"
        )
    train_index.extend([cluster_dict[cluster][0] for cluster in singleton_clusters])

    remaining_clusters = [
        cluster for cluster in cluster_dict if cluster not in singleton_clusters
    ]
    remaining_examples = len(smiles) - len(train_index)
    updated_split = test_size * len(smiles) / remaining_examples

    for cluster in remaining_clusters:
        idx_list = cluster_dict[cluster]
        clust_train, clust_test = train_test_split(
            idx_list, test_size=updated_split, random_state=random_state
        )
        train_index.extend(clust_train)
        test_index.extend(clust_test)

    smiles_tr = np.array(smiles)[train_index]
    y_tr = np.array(y)[train_index]
    smiles_te = np.array(smiles)[test_index]
    y_te = np.array(y)[test_index]
    return smiles_tr, smiles_te, y_tr, y_te


def splitter(
    smiles, y, method="random", random_state=42, test_size=0.2, n_bins=5, cutoff=0.3
):
    """
    Split a set of smiles and associated target variables by various methods.
    """
    if method.lower() == "random":
        return train_test_split(
            smiles, y, test_size=test_size, random_state=random_state
        )

    if method.lower() == "stratified":
        if n_bins is None:
            raise ValueError("n_bins must be specified for stratified split")
        return stratified_split(smiles, y, random_state, test_size, n_bins)

    if method.lower() == "scaffold":
        if cutoff is None:
            raise ValueError("cutoff must be specified for scaffold split")
        return scaffold_split(smiles, y, random_state, test_size, cutoff)

    raise ValueError(
        f"Unknown method '{method}'. Supported methods are: 'random', 'stratified', 'scaffold'"
    )


# Helper function for obtaining covariance matrices from sklearn and gpytorch GP models
# Sklearn not currently implemented

# def get_cov_matrices_sklearn(gp_model, X_train, candidates):
#     """
#     Get the covariance matrices from a scikit-learn GP model.

#     Parameters:
#         gp_model: Trained scikit-learn GaussianProcessRegressor
#         X_train: Training data (np.ndarray)
#         candidates: Candidate points for selection (np.ndarray)

#     Returns:
#         K_train_train: Covariance matrix of the training data
#         K_train_candidates: Covariance between training and candidate points
#         K_candidates_candidates: Covariance between candidate points
#     """
#     K_train_train = gp_model.kernel_(X_train)
#     K_train_candidates = gp_model.kernel_(X_train, candidates)
#     K_candidates_candidates = gp_model.kernel_(candidates)

#     return K_train_train, K_train_candidates, K_candidates_candidates


def get_cov_matrices_gpytorch(gp_model, X_train, candidates):
    """
    Get the covariance matrices from a GPyTorch GP model.

    Parameters:
        gp_model: Trained GPyTorch GP model
        X_train: Training data (torch.Tensor)
        candidates: Candidate points for selection (torch.Tensor)

    Returns:
        K_train_train: Covariance matrix of the training data (np.ndarray)
        K_train_candidates: Covariance between training and candidate points (np.ndarray)
        K_candidates_candidates: Covariance between candidate points (np.ndarray)
    """
    gp_model.eval()

    K_train_train = gp_model.covar_module(X_train).to_dense().detach().numpy()
    K_train_candidates = (
        gp_model.covar_module(X_train, candidates).to_dense().detach().numpy()
    )
    K_candidates_candidates = (
        gp_model.covar_module(candidates).to_dense().detach().numpy()
    )

    return K_train_train, K_train_candidates, K_candidates_candidates
