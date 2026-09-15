"""
Conversation Analyzer Service.
Provides enterprise session analytics, stability segments, event tallies, and token estimations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ConversationAnalyzer:
    """Computes aggregate analytics, event metrics, and telemetry summaries."""

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
                "expansion_turns": 0,
                "warning_turns": 0,
                "critical_turns": 0,
                "recoveries_count": 0,
                "topic_switches_count": 0,
                "longest_stable_segment": 0,
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
        events = df["event"] if "event" in df else pd.Series(["Normal"])

        normal_count = int((events == "Normal").sum())
        expansion_count = int((events == "Expansion").sum())
        recovery_count = int((events == "Recovery").sum())
        critical_drift_count = int((events == "Critical Drift").sum())

        warning_status_count = int((statuses == "WARNING").sum())
        critical_status_count = int((statuses == "CRITICAL").sum())

        # Longest stable segment (consecutive turns of Normal or Expansion)
        max_stable = 0
        current_stable = 0
        topic_switches = 0
        prev_is_critical = False

        for idx, ev in enumerate(events):
            if ev in ["Normal", "Expansion"]:
                current_stable += 1
                max_stable = max(max_stable, current_stable)
                prev_is_critical = False
            elif ev == "Critical Drift":
                current_stable = 0
                if not prev_is_critical:
                    topic_switches += 1
                prev_is_critical = True
            else:
                current_stable = 0
                prev_is_critical = False

        drifted_total = warning_status_count + critical_status_count
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
            "expansion_turns": expansion_count,
            "warning_turns": warning_status_count,
            "critical_turns": critical_status_count,
            "recoveries_count": recovery_count,
            "topic_switches_count": topic_switches,
            "longest_stable_segment": max_stable,
            "drift_percentage": drift_pct,
            "approx_token_count": approx_tokens,
        }
