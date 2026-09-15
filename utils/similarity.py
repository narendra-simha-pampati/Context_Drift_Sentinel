"""
Mathematical and vector similarity utilities for Context Drift Sentinel.
Supports multi-anchor semantic similarity, calibrated drift scoring, and event classification.
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


def calculate_multi_anchor_similarity(
    s_initial: float,
    s_rolling: float,
    s_summary: float,
    w_initial: float = 0.4,
    w_rolling: float = 0.4,
    w_summary: float = 0.2,
) -> float:
    """
    Compute multi-anchor composite semantic similarity:
    Final Similarity = 0.4 * Initial Intent + 0.4 * Rolling Context + 0.2 * Conversation Summary
    """
    sim = (w_initial * s_initial) + (w_rolling * s_rolling) + (w_summary * s_summary)
    return float(np.clip(sim, -1.0, 1.0))


def calculate_calibrated_drift_score(
    s_initial: float,
    s_rolling: float,
    s_summary: float,
    s_composite: float,
) -> float:
    """
    Calibrate composite similarity into an enterprise 0-100 Drift Score.
    Expected Behavior:
      - Stable technical dialogue (follow-ups, testing, details): 5 - 20
      - Minor topic expansion: 20 - 35
      - Gradual drift: 40 - 60
      - Complete topic change: 80 - 100
    """
    cohesion = 0.5 * s_rolling + 0.5 * s_summary

    # 1. Stable technical conversation (domain integrity preserved)
    if s_initial >= 0.22 and cohesion >= 0.45:
        drift = np.interp(s_composite, [0.42, 0.82], [20.0, 5.0])
    # 2. Minor topic expansion (sub-topic or related tooling)
    elif s_initial >= 0.16 or cohesion >= 0.40:
        drift = np.interp(s_composite, [0.32, 0.50], [35.0, 20.0])
    # 3. Gradual drift (moving to adjacent operational/billing topics)
    elif s_initial >= 0.10 or cohesion >= 0.25:
        drift = np.interp(s_composite, [0.20, 0.35], [60.0, 40.0])
    # 4. Complete topic switch (unrelated domain)
    else:
        drift = np.interp(s_composite, [0.05, 0.20], [100.0, 80.0])

    return float(np.clip(drift, 0.0, 100.0))


def calculate_drift_score(similarity: float) -> float:
    """Backward-compatible drift score calculation from single similarity value."""
    if similarity >= 0.60:
        drift = np.interp(similarity, [0.60, 0.90], [20.0, 5.0])
    elif similarity >= 0.40:
        drift = np.interp(similarity, [0.40, 0.60], [35.0, 20.0])
    elif similarity >= 0.20:
        drift = np.interp(similarity, [0.20, 0.40], [60.0, 35.0])
    else:
        drift = np.interp(similarity, [0.05, 0.20], [100.0, 60.0])
    return float(np.clip(drift, 0.0, 100.0))


def classify_drift_status(
    drift_score: float,
    warning_drift: float = 35.0,
    critical_drift: float = 60.0,
) -> str:
    """Classify drift into 'NORMAL', 'WARNING', or 'CRITICAL' based on drift score."""
    if drift_score < warning_drift:
        return "NORMAL"
    elif drift_score < critical_drift:
        return "WARNING"
    else:
        return "CRITICAL"


def classify_turn_event(
    turn_index: int,
    current_drift: float,
    prev_drift: float,
    s_initial: float,
    s_summary: float,
) -> str:
    """
    Classify turn into semantic event:
      - 'Normal': On-topic, aligned (Drift 0 - 20)
      - 'Expansion': Minor sub-topic / implementation expansion (Drift 20 - 35)
      - 'Recovery': Returning back to primary intent after elevated drift
      - 'Critical Drift': Severe divergence or complete topic switch (Drift >= 60)
    """
    if turn_index == 0:
        return "Normal"

    # Detect Recovery: previously drifted (>35), current drift drops by >= 3.0 points, and intent similarity rebounds
    if prev_drift > 35.0 and current_drift < (prev_drift - 3.0) and (s_initial >= 0.25 or s_summary >= 0.35):
        return "Recovery"

    # Detect Critical Drift / Topic Switch
    if current_drift >= 60.0 or (s_initial < 0.10 and s_summary < 0.15):
        return "Critical Drift"

    # Detect Topic Expansion
    if current_drift >= 20.0:
        return "Expansion"

    return "Normal"


def compute_exponential_moving_average(values: List[float], alpha: float = 0.35) -> List[float]:
    """Calculate exponential moving average to smooth noisy conversational drift curves."""
    if not values:
        return []
    ema: List[float] = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1.0 - alpha) * ema[-1])
    return ema
