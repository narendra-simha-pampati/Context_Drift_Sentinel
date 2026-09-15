"""
Enterprise Plotly chart generators for Context Drift Sentinel.
Clean, modern, and minimal design styled for AI engineering tooling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# Professional enterprise color palette
PALETTE = {
    "normal": "#10b981",    # Emerald green
    "expansion": "#3b82f6", # Blue
    "warning": "#f59e0b",   # Amber
    "critical": "#ef4444",  # Crimson
    "recovery": "#8b5cf6",  # Purple
    "primary": "#3b82f6",   # Blue
    "user": "#6366f1",      # Indigo
    "assistant": "#06b6d4", # Cyan
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(156, 163, 175, 0.2)",
    "text": "#0f172a",
}


def create_drift_timeline_chart(
    messages: List[Dict[str, Any]],
    warning_threshold: float = 35.0,   # Drift score threshold
    critical_threshold: float = 60.0,  # Drift score threshold
    show_ema: bool = True,
    chart_metric: str = "similarity",  # "similarity" or "drift"
) -> go.Figure:
    """
    Build interactive Plotly timeline chart showing conversational similarity/drift across turns.
    Includes threshold bands, markers, tooltips, inflection callouts, and recovery annotations.
    """
    if not messages:
        fig = go.Figure()
        fig.update_layout(title="No conversation data available")
        return fig

    turns = [m.get("turn_index", i) for i, m in enumerate(messages)]
    sims = [float(m.get("similarity_score", 1.0)) for m in messages]
    drifts = [float(m.get("drift_score", 0.0)) for m in messages]
    roles = [m.get("role", "unknown") for m in messages]
    statuses = [m.get("drift_status", "NORMAL") for m in messages]
    events = [m.get("event", "Normal") for m in messages]
    snippets = [
        (m.get("content", "")[:90] + "...") if len(m.get("content", "")) > 90 else m.get("content", "")
        for m in messages
    ]

    is_sim = chart_metric == "similarity"
    y_values = sims if is_sim else drifts
    y_title = "Multi-Anchor Semantic Similarity" if is_sim else "Drift Score (0-100)"

    # Event / Status color mapping
    colors = []
    for ev, st in zip(events, statuses):
        if ev == "Recovery":
            colors.append(PALETTE["recovery"])
        elif st == "CRITICAL" or ev == "Critical Drift":
            colors.append(PALETTE["critical"])
        elif st == "WARNING" or ev == "Expansion":
            colors.append(PALETTE["warning"])
        else:
            colors.append(PALETTE["normal"])

    fig = go.Figure()
    max_turn = max(turns) if turns else 1

    # Add shaded threshold background zones
    if not is_sim:
        # Green Zone (Normal: 0 - warning_threshold)
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=max_turn + 0.5,
            y0=-5,
            y1=warning_threshold,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line=dict(width=0),
            layer="below",
        )
        # Yellow Zone (Warning/Expansion: warning_threshold - critical_threshold)
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=max_turn + 0.5,
            y0=warning_threshold,
            y1=critical_threshold,
            fillcolor="rgba(245, 158, 11, 0.08)",
            line=dict(width=0),
            layer="below",
        )
        # Red Zone (Critical: critical_threshold - 105)
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=max_turn + 0.5,
            y0=critical_threshold,
            y1=105,
            fillcolor="rgba(239, 68, 68, 0.08)",
            line=dict(width=0),
            layer="below",
        )

        # Threshold lines
        fig.add_hline(
            y=warning_threshold,
            line_dash="dash",
            line_color="rgba(245, 158, 11, 0.8)",
            line_width=1.5,
            annotation_text=f"Warning Level ({warning_threshold:.0f})",
            annotation_position="bottom right",
            annotation_font_size=11,
        )
        fig.add_hline(
            y=critical_threshold,
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.8)",
            line_width=1.5,
            annotation_text=f"Critical Drift ({critical_threshold:.0f})",
            annotation_position="top right",
            annotation_font_size=11,
        )
    else:
        # Similarity mode lines
        fig.add_hline(
            y=0.55,
            line_dash="dash",
            line_color="rgba(245, 158, 11, 0.7)",
            line_width=1.5,
            annotation_text="High Alignment Threshold (0.55)",
            annotation_position="top right",
            annotation_font_size=11,
        )
        fig.add_hline(
            y=0.35,
            line_dash="dot",
            line_color="rgba(239, 68, 68, 0.7)",
            line_width=1.5,
            annotation_text="Drift Breach Threshold (0.35)",
            annotation_position="bottom right",
            annotation_font_size=11,
        )

    # Main connection line
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=y_values,
            mode="lines",
            line=dict(color="#64748b", width=2.5),
            hoverinfo="skip",
            showlegend=False,
            name="Trajectory",
        )
    )

    # Markers for each turn
    hover_templates = [
        f"<b>Turn {t}</b> ({r.upper()})<br>"
        f"Semantic Sim: <b>{s:.3f}</b><br>"
        f"Drift Score: <b>{d:.1f}/100</b><br>"
        f"Event: <b>{ev}</b> | Status: {st}<br>"
        f"<i>Text: {snip}</i><extra></extra>"
        for t, r, s, d, st, ev, snip in zip(turns, roles, sims, drifts, statuses, events, snippets)
    ]

    fig.add_trace(
        go.Scatter(
            x=turns,
            y=y_values,
            mode="markers",
            marker=dict(
                size=11,
                color=colors,
                line=dict(width=2, color="#ffffff"),
                symbol=["circle" if r == "user" else "diamond" for r in roles],
            ),
            text=hover_templates,
            hoverinfo="text",
            name="Turns",
            showlegend=False,
        )
    )

    # Detect and annotate first drift inflection turn
    drift_start_turn: Optional[int] = None
    for idx, (d, st) in enumerate(zip(drifts, statuses)):
        if idx > 0 and (st in ["WARNING", "CRITICAL"] or d >= warning_threshold):
            drift_start_turn = turns[idx]
            break

    if drift_start_turn is not None and len(turns) > drift_start_turn:
        val = y_values[drift_start_turn]
        fig.add_annotation(
            x=drift_start_turn,
            y=val,
            text="⚠️ Drift Origin",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#f59e0b",
            ax=0,
            ay=-38,
            bgcolor="#ffffff",
            bordercolor="#f59e0b",
            borderwidth=1.5,
            borderpad=4,
            font=dict(size=11, color="#b45309", family="sans-serif"),
        )

    # Detect and annotate first recovery turn (if any)
    recovery_turn: Optional[int] = None
    for idx, ev in enumerate(events):
        if ev == "Recovery":
            recovery_turn = turns[idx]
            break

    if recovery_turn is not None and len(turns) > recovery_turn:
        rec_val = y_values[recovery_turn]
        fig.add_annotation(
            x=recovery_turn,
            y=rec_val,
            text="🔄 Topic Recovery",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#8b5cf6",
            ax=0,
            ay=38,
            bgcolor="#ffffff",
            bordercolor="#8b5cf6",
            borderwidth=1.5,
            borderpad=4,
            font=dict(size=11, color="#6d28d9", family="sans-serif"),
        )

    fig.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(
            title="Conversation Turn #",
            dtick=1 if len(turns) < 25 else None,
            gridcolor=PALETTE["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            range=[-0.05, 1.05] if is_sim else [-5, 105],
            gridcolor=PALETTE["grid"],
            zeroline=False,
        ),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor=PALETTE["bg"],
        height=380,
        hovermode="closest",
    )

    return fig


def create_drift_distribution_chart(messages: List[Dict[str, Any]]) -> go.Figure:
    """Histogram showing the distribution of drift scores across conversation turns."""
    if not messages:
        return go.Figure()

    drifts = [float(m.get("drift_score", 0.0)) for m in messages]
    roles = [m.get("role", "unknown") for m in messages]

    df = pd.DataFrame({"Drift Score": drifts, "Role": roles})

    fig = px.histogram(
        df,
        x="Drift Score",
        color="Role",
        nbins=15,
        opacity=0.8,
        color_discrete_map={"user": PALETTE["user"], "assistant": PALETTE["assistant"]},
        barmode="overlay",
    )

    fig.update_layout(
        margin=dict(l=30, r=30, t=20, b=30),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor=PALETTE["bg"],
        height=240,
        xaxis=dict(title="Drift Score (0=Aligned, 100=Drifted)", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Message Count", gridcolor=PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_role_drift_comparison(messages: List[Dict[str, Any]]) -> go.Figure:
    """Bar chart comparing User vs Assistant average and maximum drift."""
    if not messages:
        return go.Figure()

    df = pd.DataFrame(messages)
    if "role" not in df.columns or "drift_score" not in df.columns:
        return go.Figure()

    agg = df.groupby("role")["drift_score"].agg(["mean", "max", "count"]).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=agg["role"],
            y=agg["mean"],
            name="Avg Drift",
            marker_color=PALETTE["primary"],
            text=[f"{v:.1f}" for v in agg["mean"]],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=agg["role"],
            y=agg["max"],
            name="Max Drift",
            marker_color=PALETTE["critical"],
            text=[f"{v:.1f}" for v in agg["max"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        margin=dict(l=30, r=30, t=20, b=30),
        plot_bgcolor="rgba(248, 250, 252, 0.5)",
        paper_bgcolor=PALETTE["bg"],
        height=240,
        barmode="group",
        xaxis=dict(title="Speaker Role", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="Score (0-100)", range=[0, 105], gridcolor=PALETTE["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
