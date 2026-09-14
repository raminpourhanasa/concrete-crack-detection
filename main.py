"""
End-to-end pipeline reproducing the method in:

  Pourhanasa, R. & Monadipour, A. (2025). "Concrete crack detection via
  graph representation learning and texture analysis."
  Innovative Infrastructure Solutions, 10:382.

Pipeline (Fig. 1):
  Annotated Images -> AGCWD preprocessing -> GLCM feature extraction ->
  Graph construction (Eq. 1) -> GCN classification (Eq. 2) -> BHO tuning

Expected data layout (e.g. the METU concrete crack dataset used in the paper):

  data/
    Positive/*.jpg
    Negative/*.jpg

Usage:
  python main.py --data_dir data --out_dir outputs --bho_calls 30
"""

import argparse
import glob
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.agcwd import preprocess_dataset
from src.glcm_features import extract_dataset_features
from src.graph_construction import build_knn_graph
from src.gcn_model import train_gcn
from src.bho_tuning import run_bho


def load_dataset(data_dir: str):
    pos = sorted(glob.glob(os.path.join(data_dir, "Positive", "*")))
    neg = sorted(glob.glob(os.path.join(data_dir, "Negative", "*")))
    paths = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    return paths, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--k_neighbors", type=int, default=8)
    ap.add_argument("--bho_calls", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    preproc_dir = os.path.join(args.out_dir, "preprocessed")

    # 1) Load raw dataset
    paths, labels = load_dataset(args.data_dir)
    print(f"Loaded {len(paths)} images ({sum(labels)} positive / "
          f"{len(labels) - sum(labels)} negative)")

    # 2) AGCWD preprocessing
    print("Running AGCWD preprocessing...")
    preprocess_dataset(paths, preproc_dir)
    preproc_paths = [os.path.join(preproc_dir, os.path.basename(p)) for p in paths]

    # 3) GLCM feature extraction
    print("Extracting GLCM texture features...")
    records = extract_dataset_features(preproc_paths, labels)

    # 4) BHO: tune Cf (Eq. 1) + GCN hyperparameters, fitness = mean 10-fold F1
    print("Running Bayesian hyperparameter optimization...")
    best = run_bho(
        records,
        n_calls=args.bho_calls,
        k_neighbors=args.k_neighbors,
        epochs=args.epochs,
        device=args.device,
    )
    print("Best hyperparameters found:", best)

    Cf = {
        "contrast": best["Cf_contrast"],
        "dissimilarity": best["Cf_dissimilarity"],
        "homogeneity": best["Cf_homogeneity"],
        "energy": best["Cf_energy"],
        "correlation": best["Cf_correlation"],
    }

    # 5) Final train/val/test split + graph construction with tuned Cf
    idx = np.arange(len(records))
    y = np.array([r["label"] for r in records])
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.3, stratify=y, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y[temp_idx], random_state=42
    )

    data = build_knn_graph(records, Cf, k=args.k_neighbors)

    n = len(records)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # 6) Train final GCN with best hyperparameters
    model, _ = train_gcn(
        data,
        train_mask,
        val_mask,
        hidden_channels=best["hidden_channels"],
        num_layers=best["num_layers"],
        lr=best["lr"],
        weight_decay=best["weight_decay"],
        dropout=best["dropout"],
        epochs=args.epochs,
        device=args.device,
    )

    # 7) Evaluate on held-out test set
    model.eval()
    data = data.to(args.device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_true = data.y.cpu().numpy()[test_idx]
    y_pred = preds[test_idx]
    y_prob = probs[test_idx]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    auc = roc_auc_score(y_true, y_prob)

    print("\n=== Test set results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "gcn_model.pt"))


if __name__ == "__main__":
    main()
