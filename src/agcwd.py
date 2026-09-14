"""
Adaptive Gamma Correction with Weighted Distribution (AGCWD)
Reference: Huang, Cheng & Chiu (2012), IEEE Trans. Image Process. 22(3):1032-1041.

Used in the paper as the preprocessing step (Fig. 1, "Image Preprocessing: AGCWD")
to normalize illumination and improve crack visibility before GLCM feature
extraction.
"""

import cv2
import numpy as np


def agcwd(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply AGCWD to a single image.

    Parameters
    ----------
    img : np.ndarray
        Input image, BGR or grayscale, uint8.
    alpha : float
        Controls the strength of the weighting function applied to the PDF.
        (Huang et al. suggest alpha in [0.5, 1.5]; we use 0.5 as a sane default.)

    Returns
    -------
    np.ndarray
        Contrast-enhanced image, same shape/dtype as input.
    """
    is_color = img.ndim == 3

    # Work on the luminance (V) channel so color images keep their hue/saturation.
    if is_color:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float64)
    else:
        v = img.astype(np.float64)

    # 1) Probability density function (PDF) of intensities
    hist, _ = np.histogram(v.flatten(), bins=256, range=(0, 255))
    pdf = hist / hist.sum()

    # 2) Weighted PDF: emphasize mid-range intensities, suppress extremes/noise
    pdf_max = pdf.max()
    pdf_min = pdf.min()
    pdf_w = pdf_max * np.power(
        (pdf - pdf_min) / (pdf_max - pdf_min + 1e-12), alpha
    )
    pdf_w[pdf == 0] = 0  # keep empty bins at zero

    # 3) Cumulative distribution function (CDF) of the weighted PDF
    cdf_w = np.cumsum(pdf_w) / (pdf_w.sum() + 1e-12)

    # 4) Adaptive gamma per intensity level: gamma(l) = 1 - CDF_w(l)
    gamma = 1.0 - cdf_w

    # 5) Apply per-pixel gamma correction via a lookup table
    levels = np.arange(256)
    table = 255.0 * np.power(levels / 255.0, gamma)
    table = np.clip(table, 0, 255).astype(np.uint8)

    v_corrected = table[np.clip(v, 0, 255).astype(np.uint8)]

    if is_color:
        hsv[:, :, 2] = v_corrected
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        out = v_corrected

    return out


def preprocess_dataset(image_paths, out_dir, alpha: float = 0.5):
    """Batch-apply AGCWD to a list of image paths and save results to out_dir."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        enhanced = agcwd(img, alpha=alpha)
        fname = os.path.basename(p)
        cv2.imwrite(os.path.join(out_dir, fname), enhanced)
