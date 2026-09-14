"""
Bayesian Hyperparameter Optimization (BHO).

Jointly tunes:
  - the GLCM-similarity weighting constants Cf (one per feature, Eq. 1)
  - the GCN hyperparameters (layers, hidden units, lr, dropout, weight decay)

using a Gaussian-Process surrogate (via scikit-optimize) to maximize
mean 10-fold F1-score, exactly as described in the "Bayesian
hyperparameter optimization (BHO)" section of the paper.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch

from .glcm_features import FEATURES
from .graph_construction import build_knn_graph
from .gcn_model import train_gcn

SPACE = [
    Real(0.1, 5.0, name="Cf_contrast"),
    Real(0.1, 5.0, name="Cf_dissimilarity"),
    Real(0.1, 5.0, name="Cf_homogeneity"),
    Real(0.1, 5.0, name="Cf_energy"),
    Real(0.1, 5.0, name="Cf_correlation"),
    Integer(1, 4, name="num_layers"),
    Integer(8, 128, name="hidden_channels"),
    Real(1e-4, 1e-2, "log-uniform", name="lr"),
    Real(0.0, 0.6, name="dropout"),
    Real(1e-6, 1e-2, "log-uniform", name="weight_decay"),
]


def run_bho(records, n_calls: int = 30, k_neighbors: int = 8, n_folds: int = 10,
            epochs: int = 60, device: str = "cpu", random_state: int = 42):
    """
    Run BHO to find the Cf constants + GCN hyperparameters that maximize
    mean 10-fold macro F1-score.

    Parameters
    ----------
    records : list[dict]
        Output of extract_dataset_features (GLCM features + labels).
    n_calls : int
        Number of BHO iterations (surrogate-guided evaluations).
    k_neighbors : int
        k for the k-NN graph construction.
    n_folds : int
        Number of CV folds used as the fitness function (paper uses 10).

    Returns
    -------
    dict with best hyperparameters and their mean F1-score.
    """
    labels = np.array([r["label"] for r in records])

    @use_named_args(SPACE)
    def objective(**params):
        Cf = {
            "contrast": params["Cf_contrast"],
            "dissimilarity": params["Cf_dissimilarity"],
            "homogeneity": params["Cf_homogeneity"],
            "energy": params["Cf_energy"],
            "correlation": params["Cf_correlation"],
        }

        # Graph is rebuilt per candidate Cf since edge weights depend on it.
        data = build_knn_graph(records, Cf, k=k_neighbors)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_f1s = []
        n = len(records)
        idx = np.arange(n)

        for train_idx, val_idx in skf.split(idx, labels):
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True

            _, val_f1 = train_gcn(
                data,
                train_mask,
                val_mask,
                hidden_channels=params["hidden_channels"],
                num_layers=params["num_layers"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                dropout=params["dropout"],
                epochs=epochs,
                device=device,
            )
            fold_f1s.append(val_f1)

        mean_f1 = float(np.mean(fold_f1s))
        # gp_minimize minimizes -> return negative F1
        return -mean_f1

    result = gp_minimize(
        objective,
        SPACE,
        n_calls=n_calls,
        random_state=random_state,
        verbose=True,
    )

    best_params = {dim.name: val for dim, val in zip(SPACE, result.x)}
    best_params["mean_f1"] = -result.fun
    return best_params
