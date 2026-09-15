# Context Drift Sentinel 🛡️
> **Enterprise AI Telemetry Platform for LLM Conversational Drift Detection, Alignment Telemetry, and Autonomous Recovery Steering.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange?style=flat-square)](https://www.sbert.net/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary & Problem Statement

In long, multi-turn LLM interactions (e.g., technical copilots, customer support bots, autonomous agentic workflows), conversations frequently experience **Context Drift**—a compounding divergence where intermediate tangents, implementation rabbit holes, or hallucinations pull the LLM away from the user's primary objective.

### Why Context Drift is a Critical Enterprise Problem:
1. **Goal Abandonment**: The LLM loses track of the core user problem after 15–20 turns.
2. **Context Window Token Waste**: Expensive tokens are wasted preserving irrelevant conversational history.
3. **Compounding Hallucinations**: When context strays, models begin inferring unrelated constraints and generating ungrounded responses.

**Context Drift Sentinel** is an internal enterprise observability tool that continuously computes vector semantic distance from the initial intent anchor to every dialogue turn. It operates **100% locally with zero external API costs**, generates real-time 0–100 drift indices, pinpoints the exact **inflection origin turn**, and outputs actionable **prompt recovery interventions**.

---

## 💡 Real-World Enterprise Use Cases

| Use Case | Target Audience | How Sentinel Helps |
|---|---|---|
| **LLM Ops & Chatbot Telemetry** | AI Platform Engineers | Continuously ingest and score production dialogue logs to measure conversation stability across model versions. |
| **Agentic Workflow Evaluation** | AI Researchers & Engineers | Identify the exact turn where multi-agent planning pipelines wander off-topic or hallucinate. |
| **Context Window Cost Optimization** | Cloud & FinOps Teams | Pinpoint conversational drift inflection points to trigger automated context pruning and save prompt cache tokens. |
| **Prompt Injection & Red-Teaming** | AI Security Teams | Detect conversational hijacking attempts and sudden topic shifts in customer-facing support bots. |

---

## 🔄 End-to-End System Workflow

```
 ┌────────────────────────────────────────────────────────┐
 │           1. Multi-Format Dialogue Ingestion           │
 │       (CSV Logs / JSON Schemas / Raw Text Transcripts) │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           2. Intent Anchoring & Segmentation           │
 │       Extracts Turn 0 User Intent (or Custom Anchor)   │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        3. Cached Vector Semantic Embeddings            │
 │     Sentence Transformers: all-MiniLM-L6-v2 (Local)    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │        4. Cosine Similarity & Drift Computation        │
 │         Sim ∈ [-1.0, 1.0]  ──►  Drift Score [0, 100]   │
 │         Status: NORMAL (Green) | WARNING | CRITICAL    │
 └───────────────────────────┬────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌──────────────────┐┌──────────────────┐┌──────────────────┐
│  Interactive     ││  Autonomous      ││  SQLite Fleet    │
│  Plotly Charts   ││  Recovery Engine ││  Telemetry &     │
│  & Turn Timeline ││  Prompt Steering ││  Leaderboards    │
└──────────────────┘└──────────────────┘└──────────────────┘
```

---

## 🧮 Multi-Anchor Mathematical Formulation

Rather than comparing dialogue turns against only the first user prompt (which causes false positives as technical conversations deepen), Sentinel uses a **production multi-anchor composite semantic similarity engine**:

### 1. The Three Semantic Anchors
1. **Initial User Intent Anchor ($\mathbf{e}_{\text{initial}}$, Weight: 40%)**:
   - Embeds the first meaningful user requirement.
   - $S_{\text{initial}} = \cos(\mathbf{e}_i, \mathbf{e}_{\text{initial}})$
2. **Rolling Conversation Context ($\mathbf{e}_{\text{rolling}}$, Weight: 40%)**:
   - Embeds the rolling window of the previous 4–6 dialogue turns with exponential decay weights for local conversational continuity.
   - $S_{\text{rolling}} = \cos(\mathbf{e}_i, \mathbf{e}_{\text{rolling}})$
3. **Running Conversation Summary ($\mathbf{e}_{\text{summary}}$, Weight: 20%)**:
   - Lightweight, non-LLM semantic summary vector representing the cumulative on-topic discussion so far.
   - $S_{\text{summary}} = \cos(\mathbf{e}_i, \mathbf{e}_{\text{summary}})$

### 2. Composite Semantic Similarity
$$\text{Final Similarity} = 0.40 \cdot S_{\text{initial}} + 0.40 \cdot S_{\text{rolling}} + 0.20 \cdot S_{\text{summary}}$$

### 3. Calibrated Drift Score (0–100 Scale) & EMA Smoothing
- **🟢 Stable Technical Dialogue (Avg Drift 5–20)**: Follow-up questions, implementation details, edge cases, unit tests, security configurations, and performance discussions stay within 5–20.
- **🔵 Minor Topic Expansion (Avg Drift 20–35)**: Related architectural components and tooling.
- **🟡 Gradual Drift (Avg Drift 40–60)**: Moving into adjacent cloud infrastructure, DNS, or billing.
- **🔴 Complete Topic Switch (Avg Drift 80–100)**: Changing domains (e.g., JWT $\to$ Cooking/Travel).
- **Smooth Real-time EMA**: Applies $\alpha = 0.35$ exponential moving average smoothing.

### 4. Dynamic Recovery Tracking
If the user or assistant naturally steers the conversation back to the primary intent anchor, **the drift score smoothly decreases back down** rather than permanently staying high, recording a **`🔄 Recovery`** event.

---

## 🚀 Key Features

### 1. Live Drift Sentinel & Plotly Visualizations
- Real-time KPI Metric Cards: **Health Status**, **Average Drift Score**, **Maximum Drift**, **Turn Count**, **Approximate Token Consumption**, and **Drift Origin Turn**.
- Interactive Plotly Trajectory Graph featuring shaded Normal/Warning/Critical zones, turn markers, and automatic **⚠️ Drift Origin** inflection callouts.
- Speaker Role Drift Comparison (User vs. Assistant) and Frequency Distribution histograms.

### 2. Conversation Timeline & Inspector
- Turn-by-turn dialogue inspector displaying role badges, timestamps, precise cosine similarity, drift indices, and full text expanders.
- Filter dialogue turns by speaker role (`User` / `Assistant`) or drift status (`NORMAL`, `WARNING`, `CRITICAL`).

### 3. AI Recovery Sentinel (Zero External API Required)
When drift exceeds thresholds, Sentinel's rule-based NLP engine automatically extracts topic divergence and generates a **3-Tier Recovery Plan**:
1. **Suggested User Prompt**: Ready-to-copy refocusing prompt for end users.
2. **Enterprise System Steering Directive**: Structured injection for LLM system prompts (`[SYSTEM STEERING INTERVENTION]`).
3. **Context Optimization**: Actionable turn-range pruning recommendations to reduce token spend.

### 4. Enterprise Fleet Analytics & Leaderboards
- Thread-safe SQLite database backend (`conversations.db`).
- Aggregates cross-session fleet metrics: **Fleet Average Drift**, **Peak Drift**, **Average Length**, **🏆 Most Stable Session**, and **⚠️ Most Drifted Session**.
- Session registry with reload, delete, and CSV/JSON export capabilities.

### 5. Multi-Format Support & Dynamic Sliders
- Ingests **CSV**, **JSON** (OpenAI chat schema, Anthropic transcripts), and **Raw Chat Transcripts**.
- Dynamic sensitivity sliders in the sidebar for real-time recalculation of warning and critical boundaries.

---

## 📂 Project Structure

```
Context_drift_sentinel/
├── app.py                      # Main Streamlit enterprise dashboard
├── requirements.txt            # Pinned dependencies
├── README.md                   # Complete architectural documentation
├── .gitignore                  # Git tracking exclusions
├── .streamlit/
│   └── config.toml             # Streamlit theme & UI styling configuration
├── database/
│   ├── __init__.py
│   ├── schema.sql              # SQLite database schema
│   ├── db_manager.py           # Thread-safe SQLite session manager
│   └── conversations.db        # SQLite persistence database
├── data/
│   ├── sample_chat.csv         # 112-turn multi-scenario realistic dataset
│   ├── sample_chat.json        # JSON sample dialogue
│   └── sample_raw_chat.txt     # Raw transcript sample
├── models/
│   ├── __init__.py
│   └── embedding_model.py      # Cached Sentence Transformers embedding manager
├── services/
│   ├── __init__.py
│   ├── drift_detector.py       # Intent anchoring, scoring & threshold engine
│   ├── recovery_engine.py      # Semantic topic shift extraction & recovery generator
│   └── conversation_analyzer.py # Session metrics, token estimation & health summary
├── utils/
│   ├── __init__.py
│   ├── parser.py               # Multi-format parser (CSV, JSON, Raw Text)
│   ├── similarity.py           # Cosine similarity math & moving averages
│   └── charts.py               # Enterprise Plotly chart builders
├── assets/
│   └── logo.png                # Sentinel badge asset
└── tests/
    ├── __init__.py
    └── test_sentinel.py        # Automated test suite (17/17 tests passing)
```

---

## 🛠️ Tech Stack (Strict Local Execution)

- **Language**: Python 3.11+
- **Frontend & Dashboard**: Streamlit
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Machine Learning & Math**: scikit-learn, NumPy, PyTorch
- **Data Analytics**: Pandas
- **Visualization**: Plotly
- **Database**: SQLite3
- **Test Framework**: Pytest / Unittest

*No cloud API keys, external databases, or third-party web services are required.*

---

## ⚡ Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/narendra-simha-pampati/Context_Drift_Sentinel.git
cd Context_Drift_Sentinel
```

### 2. Create and Activate Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Application
```bash
streamlit run app.py
```
The application will start immediately at **`http://localhost:8501`**.

---

## 🧪 Running Automated Tests

Context Drift Sentinel includes a comprehensive test suite covering mathematical invariants, parsing edge cases, SQLite operations, and prompt steering generation:

```bash
# Run with Pytest
pytest tests/ -v

# Or run with standard Unittest
python3 -m unittest discover -s tests -v
```

**Verification Results**: `17 passed in 0.81s (100% success rate)`

---

## 🎬 How to Demo Context Drift Sentinel (3-Minute Script)

1. **Step 1: Scenario Loading**
   - In the sidebar under **Select Pre-built Scenario**, select **`DB Migration (Slow Drift)`** and click **Load Scenario**.
2. **Step 2: Inspecting KPIs & Origin Point**
   - Point out the **Health Status: CRITICAL**, **Average Drift: 83.1/100**, and the **Drift Origin: Turn 4** indicator.
3. **Step 3: Trajectory Chart**
   - Show how the Plotly line chart plunges from the green on-topic zone into the red critical zone as the conversation shifts from PostgreSQL schema migration to Delaware C-Corps.
4. **Step 4: AI Recovery & Prompt Steering**
   - Open the **🛡️ Recovery Sentinel** tab to demonstrate how Sentinel extracted the topic shift and generated ready-to-inject User and System Steering prompts.
5. **Step 5: Stability Comparison**
   - Switch scenario to **`JWT Auth (Stable)`** and show how an aligned conversation stays consistently inside the safe threshold bands.
6. **Step 6: Fleet Telemetry**
   - Open the **📈 Enterprise Analytics** tab to show cross-session leaderboards and database persistence.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.