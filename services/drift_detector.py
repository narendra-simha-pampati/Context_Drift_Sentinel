"""
Core Context Drift Detection Service.
Multi-Anchor Semantic Drift Engine using Sentence Transformers.
Combines:
  1. Initial User Intent (40%)
  2. Rolling Conversation Context (40%)
  3. Running Conversation Summary (20%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.embedding_model import EmbeddingManager, get_embedding_manager
from utils.similarity import (
    calculate_calibrated_drift_score,
    calculate_cosine_similarity,
    calculate_multi_anchor_similarity,
    classify_drift_status,
    classify_turn_event,
)


@dataclass
class DriftAnalysisResult:
    """Encapsulates the complete multi-anchor drift analysis for a conversation."""
    reference_intent: str
    messages: List[Dict[str, Any]]
    avg_similarity: float
    avg_drift: float
    max_drift: float
    warning_threshold: float
    critical_threshold: float
    recoveries_count: int = 0
    topic_switches_count: int = 0
    longest_stable_segment: int = 0
    inflection_turn: Optional[int] = None
    overall_status: str = "NORMAL"
    drifted_turns_count: int = 0
    stability_score: float = 100.0


class DriftDetector:
    """Production-grade multi-anchor semantic drift detection engine."""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        warning_threshold: float = 35.0,   # Drift Score threshold
        critical_threshold: float = 60.0,  # Drift Score threshold
        w_initial: float = 0.40,
        w_rolling: float = 0.40,
        w_summary: float = 0.20,
        rolling_window_size: int = 5,
        ema_alpha: float = 0.35,
    ) -> None:
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.w_initial = w_initial
        self.w_rolling = w_rolling
        self.w_summary = w_summary
        self.rolling_window_size = rolling_window_size
        self.ema_alpha = ema_alpha

    def analyze(
        self,
        messages: List[Dict[str, Any]],
        custom_anchor_intent: Optional[str] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ) -> DriftAnalysisResult:
        """
        Execute multi-anchor drift analysis on dialogue turns.

        Args:
            messages: List of message dictionaries containing 'role', 'content', 'turn_index'.
            custom_anchor_intent: Optional explicit anchor prompt string.
            warning_threshold: Optional override for warning drift score threshold.
            critical_threshold: Optional override for critical drift score threshold.
        """
        w_thresh = warning_threshold if warning_threshold is not None else self.warning_threshold
        c_thresh = critical_threshold if critical_threshold is not None else self.critical_threshold

        if not messages:
            return DriftAnalysisResult(
                reference_intent=custom_anchor_intent or "",
                messages=[],
                avg_similarity=1.0,
                avg_drift=0.0,
                max_drift=0.0,
                warning_threshold=w_thresh,
                critical_threshold=c_thresh,
                recoveries_count=0,
                topic_switches_count=0,
                longest_stable_segment=0,
                inflection_turn=None,
                overall_status="NORMAL",
                drifted_turns_count=0,
                stability_score=100.0,
            )

        # 1. Determine Initial User Intent Anchor
        if custom_anchor_intent and custom_anchor_intent.strip():
            reference_intent = custom_anchor_intent.strip()
        else:
            # First meaningful user message, or first message
            user_msg = next(
                (m for m in messages if m.get("role") == "user" and len(str(m.get("content", "")).strip()) > 3),
                messages[0],
            )
            reference_intent = str(user_msg.get("content", "")).strip()

        # 2. Batch encode all message contents and anchor (cached)
        all_texts = [reference_intent] + [str(m.get("content", "")).strip() for m in messages]
        embeddings = self.embedding_manager.encode(all_texts)

        ref_embedding = embeddings[0]
        msg_embeddings = embeddings[1:]

        # 3. Multi-Anchor Sequential Evaluation
        analyzed_messages: List[Dict[str, Any]] = []
        similarities: List[float] = []
        drift_scores: List[float] = []
        events: List[str] = []

        inflection_turn: Optional[int] = None
        recoveries_count = 0
        topic_switches_count = 0

        # Running summary vector initialization (normalized representation)
        running_summary_vec = np.copy(ref_embedding)
        running_summary_vec /= (np.linalg.norm(running_summary_vec) + 1e-9)

        prev_smoothed_drift = 5.0
        current_stable_streak = 0
        max_stable_streak = 0

        for idx, msg in enumerate(messages):
            curr_vec = msg_embeddings[idx]

            # Anchor 1: Initial User Intent Similarity (Weight 0.4)
            s_initial = float(cosine_similarity(curr_vec.reshape(1, -1), ref_embedding.reshape(1, -1))[0][0])
            s_initial = np.clip(s_initial, -1.0, 1.0)

            # Anchor 2: Rolling Conversation Context (previous 4-6 turns, Weight 0.4)
            if idx == 0:
                s_rolling = 1.0
            else:
                w_start = max(0, idx - self.rolling_window_size)
                rolling_window_vecs = msg_embeddings[w_start:idx]
                # Exponential decay weighting giving higher weight to immediate previous turns
                k = len(rolling_window_vecs)
                weights = np.exp(np.linspace(0.0, 1.0, k))
                weights /= np.sum(weights)
                rolling_vec = np.sum(rolling_window_vecs * weights[:, np.newaxis], axis=0)
                rolling_vec /= (np.linalg.norm(rolling_vec) + 1e-9)
                s_rolling = float(cosine_similarity(curr_vec.reshape(1, -1), rolling_vec.reshape(1, -1))[0][0])
                s_rolling = np.clip(s_rolling, -1.0, 1.0)

            # Anchor 3: Running Conversation Summary (Weight 0.2)
            if idx == 0:
                s_summary = 1.0
            else:
                s_summary = float(cosine_similarity(curr_vec.reshape(1, -1), running_summary_vec.reshape(1, -1))[0][0])
                s_summary = np.clip(s_summary, -1.0, 1.0)

            # Composite Multi-Anchor Similarity: 0.4 * Initial + 0.4 * Rolling + 0.2 * Summary
            s_composite = calculate_multi_anchor_similarity(
                s_initial=s_initial,
                s_rolling=s_rolling,
                s_summary=s_summary,
                w_initial=self.w_initial,
                w_rolling=self.w_rolling,
                w_summary=self.w_summary,
            )

            # Calibrated Drift Score & Exponential Moving Average Smoothing
            raw_drift = calculate_calibrated_drift_score(
                s_initial=s_initial,
                s_rolling=s_rolling,
                s_summary=s_summary,
                s_composite=s_composite,
            )

            if idx == 0:
                smoothed_drift = raw_drift
            else:
                smoothed_drift = (self.ema_alpha * raw_drift) + ((1.0 - self.ema_alpha) * prev_smoothed_drift)

            # Status classification (NORMAL / WARNING / CRITICAL)
            status = classify_drift_status(
                drift_score=smoothed_drift,
                warning_drift=w_thresh,
                critical_drift=c_thresh,
            )

            # Event classification (Normal / Expansion / Recovery / Critical Drift)
            event = classify_turn_event(
                turn_index=idx,
                current_drift=smoothed_drift,
                prev_drift=prev_smoothed_drift,
                s_initial=s_initial,
                s_summary=s_summary,
            )

            # Track event tallies
            if event == "Recovery":
                recoveries_count += 1
            elif event == "Critical Drift" and (idx == 1 or events[-1] != "Critical Drift"):
                topic_switches_count += 1

            # Stable segment tracking (Normal or Expansion turns)
            if event in ["Normal", "Expansion"]:
                current_stable_streak += 1
                max_stable_streak = max(max_stable_streak, current_stable_streak)
            else:
                current_stable_streak = 0

            # Inflection Turn origin
            if status in ["WARNING", "CRITICAL"] and inflection_turn is None and idx > 0:
                inflection_turn = msg.get("turn_index", idx)

            # Update running summary vector with turns that are aligned (drift < 35)
            if smoothed_drift < 35.0:
                running_summary_vec = (0.85 * running_summary_vec) + (0.15 * curr_vec)
                running_summary_vec /= (np.linalg.norm(running_summary_vec) + 1e-9)

            prev_smoothed_drift = smoothed_drift

            similarities.append(s_composite)
            drift_scores.append(smoothed_drift)
            events.append(event)

            updated_msg = dict(msg)
            updated_msg["turn_index"] = msg.get("turn_index", idx)
            updated_msg["similarity_score"] = round(s_composite, 4)
            updated_msg["drift_score"] = round(smoothed_drift, 1)
            updated_msg["drift_status"] = status
            updated_msg["event"] = event
            updated_msg["s_initial"] = round(s_initial, 3)
            updated_msg["s_rolling"] = round(s_rolling, 3)
            updated_msg["s_summary"] = round(s_summary, 3)
            analyzed_messages.append(updated_msg)

        avg_similarity = float(np.mean(similarities)) if similarities else 1.0
        avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
        max_drift = float(np.max(drift_scores)) if drift_scores else 0.0
        drifted_count = sum(1 for m in analyzed_messages if m["drift_status"] in ["WARNING", "CRITICAL"])

        # Overall Status
        if avg_drift >= 55.0 or (sum(1 for m in analyzed_messages if m["drift_status"] == "CRITICAL") / max(len(messages), 1)) >= 0.35:
            overall_status = "CRITICAL"
        elif avg_drift >= 35.0 or drifted_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "NORMAL"

        # Stability Score (100 = completely rock-solid on-topic)
        drift_std = float(np.std(drift_scores)) if len(drift_scores) > 1 else 0.0
        stability = max(0.0, min(100.0, 100.0 - (avg_drift * 0.8 + drift_std * 0.5)))

        return DriftAnalysisResult(
            reference_intent=reference_intent,
            messages=analyzed_messages,
            avg_similarity=round(avg_similarity, 3),
            avg_drift=round(avg_drift, 1),
            max_drift=round(max_drift, 1),
            warning_threshold=w_thresh,
            critical_threshold=c_thresh,
            recoveries_count=recoveries_count,
            topic_switches_count=topic_switches_count,
            longest_stable_segment=max_stable_streak,
            inflection_turn=inflection_turn,
            overall_status=overall_status,
            drifted_turns_count=drifted_count,
            stability_score=round(stability, 1),
        )
