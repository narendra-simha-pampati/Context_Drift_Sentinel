"""
Conversation Analyzer Service.
Provides high-level session health metrics, token approximations, and trend analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ConversationAnalyzer:
    """Computes aggregate analytics and metrics for conversation sessions."""

    @staticmethod
    def compute_session_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive statistics for a list of analyzed messages."""
        if not messages:
            return {
                "total_turns": 0,
                "user_turns": 0,
                "assistant_turns": 0,
                "avg_similarity": 1.0,
                "avg_drift": 0.0,
                "max_drift": 0.0,
                "min_similarity": 1.0,
                "normal_turns": 0,
                "warning_turns": 0,
                "critical_turns": 0,
                "drift_percentage": 0.0,
                "approx_token_count": 0,
            }

        df = pd.DataFrame(messages)
        total_turns = len(messages)
        user_turns = len(df[df["role"] == "user"]) if "role" in df else 0
        assistant_turns = len(df[df["role"] == "assistant"]) if "role" in df else 0

        sims = df["similarity_score"].astype(float) if "similarity_score" in df else pd.Series([1.0])
        drifts = df["drift_score"].astype(float) if "drift_score" in df else pd.Series([0.0])
        statuses = df["drift_status"] if "drift_status" in df else pd.Series(["NORMAL"])

        normal_count = int((statuses == "NORMAL").sum())
        warning_count = int((statuses == "WARNING").sum())
        critical_count = int((statuses == "CRITICAL").sum())

        drifted_total = warning_count + critical_count
        drift_pct = round((drifted_total / total_turns) * 100.0, 1)

        # Rough token approximation (~4 chars per token)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        approx_tokens = max(1, total_chars // 4)

        return {
            "total_turns": total_turns,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "avg_similarity": round(float(sims.mean()), 3),
            "avg_drift": round(float(drifts.mean()), 1),
            "max_drift": round(float(drifts.max()), 1),
            "min_similarity": round(float(sims.min()), 3),
            "normal_turns": normal_count,
            "warning_turns": warning_count,
            "critical_turns": critical_count,
            "drift_percentage": drift_pct,
            "approx_token_count": approx_tokens,
        }
