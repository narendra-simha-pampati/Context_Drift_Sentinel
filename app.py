"""
Context Drift Sentinel
Enterprise AI Tooling Application for LLM Conversation Drift Detection & Recovery.

Run with: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from database.db_manager import DatabaseManager, get_db
from models.embedding_model import EmbeddingManager, get_embedding_manager
from services.conversation_analyzer import ConversationAnalyzer
from services.drift_detector import DriftAnalysisResult, DriftDetector
from services.recovery_engine import RecoveryEngine, RecoverySuggestion
from utils.charts import (
    create_drift_distribution_chart,
    create_drift_timeline_chart,
    create_role_drift_comparison,
)
from utils.parser import parse_conversation_file, parse_raw_text
from utils.similarity import calculate_drift_score, classify_drift_status


# Page configuration
st.set_page_config(
    page_title="Context Drift Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS for a sleek, minimal, high-tech dashboard
CUSTOM_CSS = """
<style>
    /* Global typography and palette */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Force crisp dark text on all standard Streamlit elements */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #1e293b;
    }
    
    /* Headers & Subtext */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar Styling & High Contrast */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] .stRadio label p,
    section[data-testid="stSidebar"] .stCheckbox label p,
    section[data-testid="stSidebar"] .stSelectbox label p,
    section[data-testid="stSidebar"] .stSlider label p,
    section[data-testid="stSidebar"] .stFileUploader label p,
    section[data-testid="stSidebar"] .stTextArea label p {
        color: #0f172a !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    section[data-testid="stSidebar"] .stCaption p {
        color: #64748b !important;
    }
    
    /* Streamlit Tabs Styling */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 600 !important;
        border-bottom-color: #3b82f6 !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a !important;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b !important;
        margin-top: 4px;
    }

    /* Status Badges */
    .status-badge-normal {
        background-color: #ecfdf5 !important;
        color: #065f46 !important;
        border: 1px solid #a7f3d0;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-warning {
        background-color: #fffbeb !important;
        color: #92400e !important;
        border: 1px solid #fde68a;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-critical {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Timeline message card */
    .chat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 4px solid #94a3b8;
    }
    .chat-card.normal {
        border-left-color: #10b981;
    }
    .chat-card.warning {
        border-left-color: #f59e0b;
    }
    .chat-card.critical {
        border-left-color: #ef4444;
    }
    
    .chat-role-user {
        color: #4338ca !important;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .chat-role-assistant {
        color: #0e7490 !important;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def init_services() -> Tuple[DatabaseManager, DriftDetector, RecoveryEngine]:
    """Initialize database and core analysis engines."""
    db = get_db()
    embedding_mgr = get_embedding_manager()
    detector = DriftDetector(embedding_manager=embedding_mgr)
    recovery_engine = RecoveryEngine()
    return db, detector, recovery_engine


def load_sample_dataset(scenario_name: str) -> List[Dict[str, Any]]:
    """Load pre-built scenario from data directory."""
    sample_csv = Path(__file__).resolve().parent / "data" / "sample_chat.csv"
    if not sample_csv.exists():
        return []

    df = pd.read_csv(sample_csv)
    if scenario_name == "All Scenarios":
        filtered_df = df
    elif scenario_name == "JWT Auth (Stable)":
        filtered_df = df[df["session_id"] == "jwt_auth_01"]
    elif scenario_name == "DB Migration (Slow Drift)":
        filtered_df = df[df["session_id"] == "db_migration_02"]
    elif scenario_name == "Python Profiling (Severe Drift)":
        filtered_df = df[df["session_id"] == "profiling_drift_03"]
    elif scenario_name == "Customer Support Guardrails (Prompt Drift)":
        filtered_df = df[df["session_id"] == "prompt_ops_04"]
    else:
        filtered_df = df

    messages: List[Dict[str, Any]] = []
    for idx, row in filtered_df.reset_index().iterrows():
        messages.append({
            "turn_index": idx,
            "role": str(row.get("role", "user")),
            "content": str(row.get("content", "")),
            "timestamp": str(row.get("timestamp", datetime.utcnow().isoformat())),
        })
    return messages


def seed_database_with_samples(db: DatabaseManager, detector: DriftDetector) -> None:
    """Seed sample sessions into database if empty."""
    existing = db.get_all_sessions()
    if existing:
        return

    scenarios = [
        ("JWT Auth Architecture (Stable)", "jwt_auth_01", "Implement secure JWT authentication in FastAPI"),
        ("DB Migration & Infrastructure (Slow Drift)", "db_migration_02", "Zero-downtime MySQL to PostgreSQL migration"),
        ("Python CPU Profiling (Severe Drift)", "profiling_drift_03", "Profile CPU and memory leaks in Python backend"),
        ("Support AI Guardrails (Topic Divergence)", "prompt_ops_04", "Insurance customer support deterministic guardrails"),
    ]

    sample_csv = Path(__file__).resolve().parent / "data" / "sample_chat.csv"
    if not sample_csv.exists():
        return

    df = pd.read_csv(sample_csv)
    for title, sess_id, summary in scenarios:
        sub_df = df[df["session_id"] == sess_id].reset_index(drop=True)
        if sub_df.empty:
            continue
        msgs = []
        for idx, row in sub_df.iterrows():
            msgs.append({
                "turn_index": idx,
                "role": str(row.get("role", "user")),
                "content": str(row.get("content", "")),
                "timestamp": str(row.get("timestamp", datetime.utcnow().isoformat())),
            })
        res = detector.analyze(msgs)
        db.save_session(
            session_id=sess_id,
            session_name=title,
            intent_summary=summary,
            messages=res.messages,
            drift_avg=res.avg_drift,
            max_drift=res.max_drift,
            status=res.overall_status,
        )


# Main App Execution
def main() -> None:
    db, detector, recovery_engine = init_services()
    seed_database_with_samples(db, detector)

    # Initialize Session State
    if "current_messages" not in st.session_state:
        st.session_state.current_messages = load_sample_dataset("DB Migration (Slow Drift)")
    if "session_name" not in st.session_state:
        st.session_state.session_name = "DB Migration (Slow Drift)"
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "custom_intent" not in st.session_state:
        st.session_state.custom_intent = ""

    # Top Header
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="margin: 0; font-size: 1.75rem;">🛡️ Context Drift Sentinel</h1>
                <span class="status-badge-normal" style="font-size: 0.72rem; letter-spacing: 0.05em;">SENTINEL ACTIVE</span>
            </div>
            <p style="color: #64748b; font-size: 0.88rem; margin-top: 0px; margin-bottom: 16px;">
                Enterprise conversational alignment telemetry, semantic drift tracking, and recovery steering.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with header_col2:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 6px;">
                <span style="font-size: 0.75rem; color: #64748b;">Active Session:</span><br>
                <strong style="font-size: 0.88rem; color: #0f172a;">{st.session_state.session_name[:24]}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar Navigation & Settings
    with st.sidebar:
        st.markdown("### ⚙️ Sentinel Controls")
        
        # Conversation Source Selector
        source_mode = st.radio(
            "Conversation Source",
            ["Preset Enterprise Scenarios", "Upload File (CSV / JSON / Text)", "Paste Raw Transcript"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        if source_mode == "Preset Enterprise Scenarios":
            preset_choice = st.selectbox(
                "Select Pre-built Scenario",
                [
                    "DB Migration (Slow Drift)",
                    "JWT Auth (Stable)",
                    "Python Profiling (Severe Drift)",
                    "Customer Support Guardrails (Prompt Drift)",
                ],
            )
            if st.button("Load Scenario", use_container_width=True, type="secondary"):
                st.session_state.current_messages = load_sample_dataset(preset_choice)
                st.session_state.session_name = preset_choice
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.custom_intent = ""
                st.rerun()

        elif source_mode == "Upload File (CSV / JSON / Text)":
            uploaded_file = st.file_uploader(
                "Upload Conversation Log",
                type=["csv", "json", "txt"],
                help="Supports CSV (role, content), JSON list of messages, or standard transcript format.",
            )
            if uploaded_file is not None:
                if st.button("Parse & Analyze Upload", use_container_width=True, type="primary"):
                    content_bytes = uploaded_file.read()
                    parsed_msgs, err = parse_conversation_file(content_bytes, uploaded_file.name)
                    if err:
                        st.error(err)
                    elif not parsed_msgs:
                        st.warning("No messages found in the uploaded file.")
                    else:
                        st.session_state.current_messages = parsed_msgs
                        st.session_state.session_name = uploaded_file.name.rsplit(".", 1)[0]
                        st.session_state.session_id = str(uuid.uuid4())[:8]
                        st.session_state.custom_intent = ""
                        st.success(f"Loaded {len(parsed_msgs)} turns.")
                        st.rerun()

        elif source_mode == "Paste Raw Transcript":
            raw_input = st.text_area(
                "Paste Dialogue Transcript",
                placeholder="User: How do I configure OAuth?\nAssistant: You can register credentials...\nUser: What about pizza?",
                height=150,
            )
            if st.button("Analyze Pasted Chat", use_container_width=True, type="primary"):
                if raw_input.strip():
                    parsed_msgs = parse_raw_text(raw_input)
                    if parsed_msgs:
                        st.session_state.current_messages = parsed_msgs
                        st.session_state.session_name = "Pasted Conversation"
                        st.session_state.session_id = str(uuid.uuid4())[:8]
                        st.session_state.custom_intent = ""
                        st.rerun()
                    else:
                        st.error("Could not parse messages from input text.")

        st.markdown("---")
        st.markdown("#### 🎯 Intent Anchor")
        use_custom_intent = st.checkbox("Define Custom Anchor Intent", value=bool(st.session_state.custom_intent))
        custom_anchor: Optional[str] = None
        if use_custom_intent:
            custom_anchor = st.text_input(
                "Reference Intent String",
                value=st.session_state.custom_intent,
                placeholder="e.g. Implement OAuth2 JWT authentication in FastAPI",
            )
            st.session_state.custom_intent = custom_anchor
        else:
            st.session_state.custom_intent = ""

        st.markdown("---")
        st.markdown("#### 🎚️ Sensitivity Thresholds")
        warning_thresh = st.slider(
            "Warning Similarity Threshold",
            min_value=0.15,
            max_value=0.75,
            value=0.30,
            step=0.01,
            help="Conversations with similarity below this trigger WARNING status.",
        )
        critical_thresh = st.slider(
            "Critical Similarity Threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.18,
            step=0.01,
            help="Conversations with similarity below this trigger CRITICAL status.",
        )

        st.markdown("---")
        st.markdown("#### 💾 Session Database")
        all_sessions = db.get_all_sessions()
        session_options = {f"{s['session_name']} ({s['created_at'][:16]})": s["id"] for s in all_sessions}

        if session_options:
            selected_sess_label = st.selectbox("Saved Sessions", list(session_options.keys()))
            selected_id = session_options[selected_sess_label]
            col_load, col_del = st.columns(2)
            with col_load:
                if st.button("Reload", use_container_width=True):
                    saved_sess = db.get_session_by_id(selected_id)
                    if saved_sess:
                        st.session_state.current_messages = saved_sess["messages"]
                        st.session_state.session_name = saved_sess["session_name"]
                        st.session_state.session_id = saved_sess["id"]
                        st.session_state.custom_intent = ""
                        st.rerun()
            with col_del:
                if st.button("Delete", use_container_width=True):
                    db.delete_session(selected_id)
                    st.success("Session deleted.")
                    st.rerun()
        else:
            st.caption("No saved sessions in database.")

    # Run Analysis
    with st.spinner("Calculating semantic drift telemetry..."):
        analysis_result: DriftAnalysisResult = detector.analyze(
            messages=st.session_state.current_messages,
            custom_anchor_intent=st.session_state.custom_intent,
            warning_threshold=warning_thresh,
            critical_threshold=critical_thresh,
        )
        summary = ConversationAnalyzer.compute_session_summary(analysis_result.messages)

    # Persist / Auto-save session
    if analysis_result.messages:
        db.save_session(
            session_id=st.session_state.session_id,
            session_name=st.session_state.session_name,
            intent_summary=analysis_result.reference_intent[:100],
            messages=analysis_result.messages,
            drift_avg=analysis_result.avg_drift,
            max_drift=analysis_result.max_drift,
            status=analysis_result.overall_status,
        )

    # Main Tabs
    tab_drift, tab_timeline, tab_recovery, tab_dashboard, tab_raw = st.tabs([
        "📊 Live Drift Sentinel",
        "💬 Conversation Timeline",
        "🛡️ Recovery Sentinel",
        "📈 Enterprise Analytics",
        "📋 Raw Data & Export",
    ])

    # -------------------------------------------------------------
    # TAB 1: Live Drift Sentinel
    # -------------------------------------------------------------
    with tab_drift:
        # Metrics Row
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            status_style = "normal" if analysis_result.overall_status == "NORMAL" else "warning" if analysis_result.overall_status == "WARNING" else "critical"
            badge_class = f"status-badge-{status_style}"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Health Status</div>
                    <div style="margin-top: 6px;"><span class="{badge_class}" style="font-size: 1rem;">{analysis_result.overall_status}</span></div>
                    <div class="metric-sub">Stability: {analysis_result.stability_score:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Average Drift</div>
                    <div class="metric-value">{analysis_result.avg_drift:.1f}<span style="font-size: 1rem; color: #64748b;">/100</span></div>
                    <div class="metric-sub">Avg Sim: {summary['avg_similarity']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Maximum Drift</div>
                    <div class="metric-value">{analysis_result.max_drift:.1f}<span style="font-size: 1rem; color: #64748b;">/100</span></div>
                    <div class="metric-sub">Min Sim: {summary['min_similarity']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Conversation Turns</div>
                    <div class="metric-value">{summary['total_turns']}</div>
                    <div class="metric-sub">~{summary['approx_token_count']:,} tokens</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m_col5:
            inflection_text = f"Turn {analysis_result.inflection_turn}" if analysis_result.inflection_turn is not None else "None (Aligned)"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Drift Origin</div>
                    <div class="metric-value" style="font-size: 1.3rem; padding-top: 4px;">{inflection_text}</div>
                    <div class="metric-sub">{summary['drift_percentage']}% turns drifted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Reference Intent Box
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px; margin: 12px 0;">
                <span style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Anchor Intent:</span>
                <span style="font-size: 0.9rem; color: #1e293b; margin-left: 8px;">"{analysis_result.reference_intent}"</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chart Controls & Main Plotly Chart
        chart_ctrl1, chart_ctrl2 = st.columns([3, 1])
        with chart_ctrl1:
            st.markdown("#### 📉 Conversational Drift Trajectory")
        with chart_ctrl2:
            chart_metric = st.selectbox("Plot Metric", ["similarity", "drift"], format_func=lambda x: "Semantic Similarity" if x == "similarity" else "Drift Score (0-100)", label_visibility="collapsed")

        timeline_fig = create_drift_timeline_chart(
            messages=analysis_result.messages,
            warning_threshold=warning_thresh,
            critical_threshold=critical_thresh,
            chart_metric=chart_metric,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Secondary Analytics Charts
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("##### 📊 Drift Score Distribution")
            dist_fig = create_drift_distribution_chart(analysis_result.messages)
            st.plotly_chart(dist_fig, use_container_width=True)

        with sub_col2:
            st.markdown("##### 👥 Speaker Role Drift Comparison")
            role_fig = create_role_drift_comparison(analysis_result.messages)
            st.plotly_chart(role_fig, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: Conversation Timeline & Inspector
    # -------------------------------------------------------------
    with tab_timeline:
        st.markdown("#### 💬 Turn-by-Turn Dialogue Inspector")
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            role_filter = st.multiselect("Filter by Speaker", ["user", "assistant"], default=["user", "assistant"])
        with filter_col2:
            status_filter = st.multiselect("Filter by Status", ["NORMAL", "WARNING", "CRITICAL"], default=["NORMAL", "WARNING", "CRITICAL"])

        filtered_turns = [
            m for m in analysis_result.messages
            if m.get("role") in role_filter and m.get("drift_status") in status_filter
        ]

        if not filtered_turns:
            st.info("No turns match the selected filter criteria.")
        else:
            for m in filtered_turns:
                turn_no = m.get("turn_index", 0)
                role = m.get("role", "user")
                sim = m.get("similarity_score", 1.0)
                drift = m.get("drift_score", 0.0)
                status = m.get("drift_status", "NORMAL")
                content = m.get("content", "")
                timestamp = m.get("timestamp", "")

                card_class = "normal" if status == "NORMAL" else "warning" if status == "WARNING" else "critical"
                role_class = f"chat-role-{role}" if role in ["user", "assistant"] else "chat-role-user"
                badge_class = f"status-badge-{card_class}"

                st.markdown(
                    f"""
                    <div class="chat-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <div>
                                <strong style="color: #0f172a; font-size: 0.88rem;">Turn #{turn_no}</strong>
                                <span class="{role_class}" style="margin-left: 8px;">{role}</span>
                                <span style="font-size: 0.75rem; color: #94a3b8; margin-left: 10px;">{timestamp[:19]}</span>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 0.8rem; color: #64748b;">Sim: <strong>{sim:.3f}</strong> | Drift: <strong>{drift:.1f}</strong></span>
                                <span class="{badge_class}">{status}</span>
                            </div>
                        </div>
                        <div style="font-size: 0.9rem; color: #334155; line-height: 1.5; white-space: pre-wrap;">{content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------
    # TAB 3: Recovery Sentinel
    # -------------------------------------------------------------
    with tab_recovery:
        st.markdown("#### 🛡️ AI Recovery & Steering Generator")
        
        suggestion: Optional[RecoverySuggestion] = recovery_engine.generate_recovery_plan(
            reference_intent=analysis_result.reference_intent,
            messages=analysis_result.messages,
            inflection_turn=analysis_result.inflection_turn,
        )

        if suggestion is None or analysis_result.overall_status == "NORMAL":
            st.success("✅ Conversation is strictly on-topic. No recovery steering required.")
            st.markdown(
                """
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: center;">
                    <h4 style="color: #059669; margin: 0;">Optimal Alignment Maintained</h4>
                    <p style="color: #64748b; font-size: 0.88rem; margin-top: 8px;">
                        All conversation turns remain within safe similarity boundaries relative to original user intent.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            rec_color = "#f59e0b" if suggestion.drift_level == "WARNING" else "#ef4444"
            st.markdown(
                f"""
                <div style="background: #ffffff; border-left: 5px solid {rec_color}; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #0f172a;">Detected Topic Divergence</h4>
                        <span class="status-badge-{suggestion.drift_level.lower()}">{suggestion.drift_level} DRIFT</span>
                    </div>
                    <p style="color: #334155; font-size: 0.92rem; margin: 0;">
                        Conversation appears to have shifted toward <strong>{suggestion.shifted_topic}</strong> while the original intent was centered on <strong>{suggestion.original_topic}</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### 1. Suggested User Recovery Prompt")
            st.info("Inject this prompt into the user chat interface to return to the core topic:")
            st.code(suggestion.suggested_user_prompt, language="text")

            st.markdown("##### 2. Enterprise System Steering Directive")
            st.caption("For automated LLM middleware: Prepend or inject this instruction into the system context window:")
            st.code(suggestion.system_steering_prompt, language="text")

            st.markdown("##### 3. Context Optimization Recommendation")
            st.warning(suggestion.context_pruning_recommendation)

    # -------------------------------------------------------------
    # TAB 4: Enterprise Analytics
    # -------------------------------------------------------------
    with tab_dashboard:
        st.markdown("#### 📈 Enterprise Fleet Telemetry (All Saved Sessions)")
        db_metrics = db.get_dashboard_metrics()

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Sessions</div>
                    <div class="metric-value">{db_metrics['total_sessions']}</div>
                    <div class="metric-sub">Stored in SQLite</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Fleet Avg Drift</div>
                    <div class="metric-value">{db_metrics['avg_drift_overall']:.1f}<span style="font-size: 1rem; color: #64748b;">/100</span></div>
                    <div class="metric-sub">Across all conversations</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Fleet Drift</div>
                    <div class="metric-value">{db_metrics['max_drift_overall']:.1f}<span style="font-size: 1rem; color: #64748b;">/100</span></div>
                    <div class="metric-sub">Highest recorded drift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with d_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Length</div>
                    <div class="metric-value">{db_metrics['avg_conversation_length']:.1f}</div>
                    <div class="metric-sub">Turns per conversation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Leaderboards
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("##### 🏆 Most Stable Conversation")
            stable = db_metrics.get("most_stable_session")
            if stable:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px 18px;">
                        <strong style="color: #065f46; font-size: 1rem;">{stable['session_name']}</strong>
                        <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">
                            Avg Drift: <strong>{stable['drift_avg']:.1f}</strong> | Length: <strong>{stable['turn_count']} turns</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No sessions available.")

        with l_col2:
            st.markdown("##### ⚠️ Most Drifted Conversation")
            drifted = db_metrics.get("most_drifted_session")
            if drifted:
                st.markdown(
                    f"""
                    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px 18px;">
                        <strong style="color: #991b1b; font-size: 1rem;">{drifted['session_name']}</strong>
                        <div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">
                            Max Drift: <strong>{drifted['max_drift']:.1f}</strong> | Avg Drift: <strong>{drifted['drift_avg']:.1f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No sessions available.")

        st.markdown("---")
        st.markdown("##### 📁 Stored Sessions Registry")
        raw_sessions = db.get_all_sessions()
        if raw_sessions:
            sess_df = pd.DataFrame(raw_sessions)
            sess_df = sess_df.rename(columns={
                "id": "Session ID",
                "session_name": "Session Name",
                "intent_summary": "Initial Intent",
                "turn_count": "Turns",
                "drift_avg": "Avg Drift",
                "max_drift": "Max Drift",
                "status": "Health Status",
                "created_at": "Timestamp",
            })
            st.dataframe(sess_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recorded sessions in database.")

    # -------------------------------------------------------------
    # TAB 5: Raw Data & Export
    # -------------------------------------------------------------
    with tab_raw:
        st.markdown("#### 📋 Analyzed Telemetry Data")
        if analysis_result.messages:
            df_export = pd.DataFrame(analysis_result.messages)
            st.dataframe(df_export, use_container_width=True, hide_index=True)

            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Telemetry CSV",
                    data=csv_data,
                    file_name=f"sentinel_telemetry_{st.session_state.session_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_col2:
                json_str = json.dumps(analysis_result.messages, indent=2)
                st.download_button(
                    label="📥 Download Telemetry JSON",
                    data=json_str,
                    file_name=f"sentinel_telemetry_{st.session_state.session_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
