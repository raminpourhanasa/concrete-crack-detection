# Concrete Crack Detection via Graph Representation Learning and Texture Analysis

Reference implementation of:

> Pourhanasa, R. & Monadipour, A. (2025). *Concrete crack detection via graph
> representation learning and texture analysis.* Innovative Infrastructure
> Solutions, 10:382. https://doi.org/10.1007/s41062-025-02176-7

## Pipeline

```
Annotated images
      │
      ▼
Image Preprocessing (AGCWD)         src/agcwd.py
      │
      ▼
Feature Extraction (GLCM)           src/glcm_features.py
      │
      ▼
Graph Construction (Eq. 1)          src/graph_construction.py
      │
      ▼
GCN Classification (Eq. 2)          src/gcn_model.py
      │
      ▼
BHO Hyperparameter Tuning           src/bho_tuning.py
      │
      ▼
Final Predictive Model
```

This mirrors Fig. 1 of the paper exactly:
1. **AGCWD** (Huang et al. 2012) adaptively corrects gamma per-image from a
   weighted intensity histogram, improving crack visibility under variable
   lighting/noise.
2. **GLCM** texture features — Contrast, Dissimilarity, Homogeneity, Energy,
   Correlation — are extracted from each preprocessed image.
3. **Graph construction**: each image is a node; edge weights follow Eq. (1)

   ```
   w_i,k = exp( -Σ_f Cf · (f_i − f_j)² / (f_i + f_j) )
   ```

   with per-feature constants `Cf`. A k-NN graph is built for tractability
   at dataset scale (the paper's Fig. 4 six-node example corresponds to
   `build_full_graph`, the exact pairwise version).
4. **GCN classification** implements the symmetric-normalized propagation
   rule of Eq. (2) via PyTorch Geometric's `GCNConv`.
5. **BHO** (Bayesian Hyperparameter Optimization, Gaussian-process surrogate)
   jointly tunes the `Cf` constants and the GCN's hyperparameters (layers,
   hidden units, learning rate, dropout, weight decay) to maximize mean
   10-fold macro F1-score, as described in the paper.

## Data

The paper uses the METU concrete crack dataset (Özgenel & Sorguç, 2018):
40,000 RGB images, 227×227, split into `Positive/` (crack) and `Negative/`
(no crack) folders. Point `--data_dir` at a directory with that structure:

```
data/
  Positive/*.jpg
  Negative/*.jpg
```

## Usage

```bash
pip install -r requirements.txt
python main.py --data_dir data --out_dir outputs --bho_calls 30 --epochs 100
```

Key flags:
- `--k_neighbors`: k for the similarity graph (default 8)
- `--bho_calls`: number of BHO surrogate evaluations (paper explores the
  space more exhaustively; reduce for quick local runs)
- `--epochs`: GCN training epochs per fold/final fit

## Notes on scaling

Building the *exact* full pairwise graph from Eq. (1) is O(n²) and
infeasible at n=40,000, so this implementation uses a k-nearest-neighbor
graph scored with the same Eq. (1) similarity function — the standard way
to realize this construction at scale. `build_full_graph` in
`src/graph_construction.py` gives the exact version for small subsets
(e.g. to reproduce the Fig. 4 illustration).

## Repository structure

```
├── main.py                  # end-to-end pipeline
├── src/
│   ├── agcwd.py              # AGCWD preprocessing
│   ├── glcm_features.py      # GLCM texture feature extraction
│   ├── graph_construction.py # Eq. 1 similarity graph
│   ├── gcn_model.py          # Eq. 2 GCN classifier
│   └── bho_tuning.py         # Bayesian hyperparameter optimization
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{pourhanasa2025crack,
  title={Concrete crack detection via graph representation learning and texture analysis},
  author={Pourhanasa, Ramin and Monadipour, Ali},
  journal={Innovative Infrastructure Solutions},
  volume={10},
  number={382},
  year={2025},
  publisher={Springer},
  doi={10.1007/s41062-025-02176-7}
}
```
