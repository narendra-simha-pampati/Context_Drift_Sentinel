-- Context Drift Sentinel Database Schema

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    intent_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    turn_count INTEGER DEFAULT 0,
    drift_avg REAL DEFAULT 0.0,
    max_drift REAL DEFAULT 0.0,
    status TEXT DEFAULT 'NORMAL'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT,
    similarity_score REAL DEFAULT 1.0,
    drift_score REAL DEFAULT 0.0,
    drift_status TEXT DEFAULT 'NORMAL',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(session_id, turn_index);
