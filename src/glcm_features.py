"""
Gray-Level Co-occurrence Matrix (GLCM) texture feature extraction.

Extracts the five features used in the paper: Contrast, Dissimilarity,
Homogeneity, Energy, Correlation. These become the per-image node
attributes used both for graph-edge similarity (Eq. 1) and as GCN
node features.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

FEATURES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]


def extract_glcm_features(
    img: np.ndarray,
    distances=(1,),
    angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    levels: int = 256,
) -> dict:
    """
    Compute GLCM texture features for one image.

    Angles/distances are averaged over to give a single, rotation-robust
    value per feature (standard practice for GLCM texture description).

    Returns
    -------
    dict mapping feature name -> float value
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    glcm = graycomatrix(
        gray,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for name in FEATURES:
        # skimage returns one value per (distance, angle); average them
        feats[name] = float(np.mean(graycoprops(glcm, name)))
    return feats


def extract_dataset_features(image_paths, labels):
    """
    Run extract_glcm_features over a dataset.

    Returns
    -------
    list[dict] with keys: path, label, contrast, dissimilarity, homogeneity,
    energy, correlation
    """
    records = []
    for path, label in zip(image_paths, labels):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        feats = extract_glcm_features(img)
        feats["path"] = path
        feats["label"] = label
        records.append(feats)
    return records
