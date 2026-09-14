"""
Graph construction from GLCM features.

Implements the similarity weight of Eq. (1) in the paper:

    w_i,k = exp( -sum_f Cf * (f_i - f_j)^2 / (f_i + f_j) )

where f ranges over {contrast, dissimilarity, homogeneity, energy,
correlation}, and Cf are per-feature hyperparameters tuned via BHO
(see bho_tuning.py).

Building the *full* pairwise graph is O(n^2) and infeasible at n=40,000,
so we build a k-nearest-neighbor graph (by the same similarity score),
which is the standard, scalable way to realize this construction.
"""

from itertools import combinations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .glcm_features import FEATURES


def pairwise_similarity(fi: dict, fj: dict, Cf: dict) -> float:
    """Eq. (1): similarity weight between two feature dicts."""
    dissimilarity = 0.0
    for f in FEATURES:
        a, b = fi[f], fj[f]
        denom = a + b
        if abs(denom) < 1e-12:
            continue
        dissimilarity += Cf[f] * ((a - b) ** 2) / denom
    return float(np.exp(-dissimilarity))


def build_knn_graph(records: list, Cf: dict, k: int = 8) -> Data:
    """
    Build a k-NN similarity graph over the dataset.

    Parameters
    ----------
    records : list[dict]
        Output of extract_dataset_features: one dict per image with GLCM
        features + 'label'.
    Cf : dict
        Per-feature weighting constants {feature_name: value}, tuned by BHO.
    k : int
        Number of neighbors per node.

    Returns
    -------
    torch_geometric.data.Data
        Graph with x = GLCM feature matrix, edge_index/edge_weight from
        Eq. (1), y = labels.
    """
    n = len(records)
    X = np.array([[r[f] for f in FEATURES] for r in records], dtype=np.float64)
    y = np.array([r["label"] for r in records], dtype=np.int64)

    # Use Euclidean NN search in feature space to shortlist candidate
    # neighbors, then score exactly with Eq. (1).
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    _, neigh_idx = nn.kneighbors(X)

    src, dst, weights = [], [], []
    for i in range(n):
        for j in neigh_idx[i]:
            if i == j:
                continue
            w = pairwise_similarity(records[i], records[j], Cf)
            src.append(i)
            dst.append(j)
            weights.append(w)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    x = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y_t)


def build_full_graph(records: list, Cf: dict) -> Data:
    """
    Build the exact full pairwise graph (Eq. 1 over every pair).
    Only practical for small subsets (e.g. the 6-node illustration in Fig. 4).
    """
    n = len(records)
    X = np.array([[r[f] for f in FEATURES] for r in records], dtype=np.float64)
    y = np.array([r["label"] for r in records], dtype=np.int64)

    src, dst, weights = [], [], []
    for i, j in combinations(range(n), 2):
        w = pairwise_similarity(records[i], records[j], Cf)
        src += [i, j]
        dst += [j, i]
        weights += [w, w]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    x = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y_t)
