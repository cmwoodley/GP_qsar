# Importing modules
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


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
