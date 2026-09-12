"""
Mathematical and vector similarity utilities for Context Drift Sentinel.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(
    reference_vector: np.ndarray,
    target_vectors: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between reference vector and an array of target vectors.

    Args:
        reference_vector: 1D or 2D array representing original intent embedding.
        target_vectors: 2D array of message embeddings.

    Returns:
        1D numpy array of cosine similarity scores in range [-1.0, 1.0].
    """
    if reference_vector.ndim == 1:
        ref = reference_vector.reshape(1, -1)
    else:
        ref = reference_vector

    if target_vectors.ndim == 1:
        targets = target_vectors.reshape(1, -1)
    else:
        targets = target_vectors

    sims = cosine_similarity(ref, targets)[0]
    return np.clip(sims, -1.0, 1.0)


def calculate_drift_score(similarity: float) -> float:
    """
    Convert a cosine similarity score [-1.0, 1.0] into a Drift Score [0, 100].
    0 = completely on topic (similarity = 1.0)
    100 = severe drift (similarity <= 0.0)
    """
    # Map similarity from [1.0, 0.0] to [0.0, 100.0]
    # If similarity is negative, cap at 100.0
    drift = (1.0 - similarity) * 100.0
    return float(np.clip(drift, 0.0, 100.0))


def classify_drift_status(
    similarity: float,
    warning_threshold: float = 0.30,
    critical_threshold: float = 0.18,
) -> str:
    """
    Classify drift into 'NORMAL', 'WARNING', or 'CRITICAL' based on similarity threshold.

    Args:
        similarity: Cosine similarity score (higher = closer to intent).
        warning_threshold: Below this triggers 'WARNING'.
        critical_threshold: Below this triggers 'CRITICAL'.

    Returns:
        Status label: 'NORMAL' | 'WARNING' | 'CRITICAL'
    """
    if similarity >= warning_threshold:
        return "NORMAL"
    elif similarity >= critical_threshold:
        return "WARNING"
    else:
        return "CRITICAL"


def compute_exponential_moving_average(values: List[float], alpha: float = 0.3) -> List[float]:
    """Calculate exponential moving average to smooth noisy conversational drift curves."""
    if not values:
        return []
    ema: List[float] = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema
