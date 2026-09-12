"""
Core Context Drift Detection Service.
Computes semantic distance between reference intent and dialogue turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from models.embedding_model import EmbeddingManager, get_embedding_manager
from utils.similarity import calculate_cosine_similarity, calculate_drift_score, classify_drift_status


@dataclass
class DriftAnalysisResult:
    """Encapsulates the complete drift analysis for a conversation."""
    reference_intent: str
    messages: List[Dict[str, Any]]
    avg_drift: float
    max_drift: float
    warning_threshold: float
    critical_threshold: float
    inflection_turn: Optional[int] = None
    overall_status: str = "NORMAL"
    drifted_turns_count: int = 0
    stability_score: float = 100.0  # 100 = completely stable, 0 = highly volatile


class DriftDetector:
    """Analyzes conversation messages against an anchor intent to detect drift."""

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        warning_threshold: float = 0.30,
        critical_threshold: float = 0.18,
    ) -> None:
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def analyze(
        self,
        messages: List[Dict[str, Any]],
        custom_anchor_intent: Optional[str] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ) -> DriftAnalysisResult:
        """
        Execute full drift analysis on conversation turns.

        Args:
            messages: List of message dictionaries with 'role', 'content', 'turn_index'.
            custom_anchor_intent: Optional explicit intent string. If None, uses Turn 0 or first user turn.
            warning_threshold: Optional override for warning similarity threshold.
            critical_threshold: Optional override for critical similarity threshold.
        """
        w_thresh = warning_threshold if warning_threshold is not None else self.warning_threshold
        c_thresh = critical_threshold if critical_threshold is not None else self.critical_threshold

        if not messages:
            return DriftAnalysisResult(
                reference_intent=custom_anchor_intent or "",
                messages=[],
                avg_drift=0.0,
                max_drift=0.0,
                warning_threshold=w_thresh,
                critical_threshold=c_thresh,
                inflection_turn=None,
                overall_status="NORMAL",
                drifted_turns_count=0,
                stability_score=100.0,
            )

        # Determine reference intent
        if custom_anchor_intent and custom_anchor_intent.strip():
            reference_intent = custom_anchor_intent.strip()
        else:
            # Find first user message, or first message
            user_msg = next((m for m in messages if m.get("role") == "user"), messages[0])
            reference_intent = user_msg.get("content", "").strip()

        # Batch encode reference and all message contents
        all_texts = [reference_intent] + [m.get("content", "") for m in messages]
        embeddings = self.embedding_manager.encode(all_texts)

        ref_embedding = embeddings[0]
        msg_embeddings = embeddings[1:]

        # Calculate cosine similarities
        sims = calculate_cosine_similarity(ref_embedding, msg_embeddings)

        analyzed_messages: List[Dict[str, Any]] = []
        drift_scores: List[float] = []
        inflection_turn: Optional[int] = None
        drifted_count = 0

        for idx, (msg, sim) in enumerate(zip(messages, sims)):
            sim_float = float(sim)
            drift_score = calculate_drift_score(sim_float)
            status = classify_drift_status(
                sim_float,
                warning_threshold=w_thresh,
                critical_threshold=c_thresh,
            )

            drift_scores.append(drift_score)

            if status in ["WARNING", "CRITICAL"]:
                drifted_count += 1
                if inflection_turn is None and idx > 0:
                    inflection_turn = msg.get("turn_index", idx)

            updated_msg = dict(msg)
            updated_msg["turn_index"] = msg.get("turn_index", idx)
            updated_msg["similarity_score"] = round(sim_float, 4)
            updated_msg["drift_score"] = round(drift_score, 2)
            updated_msg["drift_status"] = status
            analyzed_messages.append(updated_msg)

        avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
        max_drift = float(np.max(drift_scores)) if drift_scores else 0.0

        # Determine overall conversation health status based on aggregate distribution
        critical_count = sum(1 for m in analyzed_messages if m["drift_status"] == "CRITICAL")
        warning_count = sum(1 for m in analyzed_messages if m["drift_status"] == "WARNING")
        total_m = max(len(messages), 1)

        crit_ratio = critical_count / total_m
        drift_ratio = (critical_count + warning_count) / total_m

        if crit_ratio >= 0.20 or avg_drift >= 70.0:
            overall_status = "CRITICAL"
        elif drift_ratio >= 0.20 or avg_drift >= 45.0:
            overall_status = "WARNING"
        else:
            overall_status = "NORMAL"

        # Stability score (100 minus penalty for volatility and high drift)
        drift_std = float(np.std(drift_scores)) if len(drift_scores) > 1 else 0.0
        stability = max(0.0, min(100.0, 100.0 - (avg_drift * 0.7 + drift_std * 1.5)))

        return DriftAnalysisResult(
            reference_intent=reference_intent,
            messages=analyzed_messages,
            avg_drift=round(avg_drift, 2),
            max_drift=round(max_drift, 2),
            warning_threshold=w_thresh,
            critical_threshold=c_thresh,
            inflection_turn=inflection_turn,
            overall_status=overall_status,
            drifted_turns_count=drifted_count,
            stability_score=round(stability, 1),
        )
