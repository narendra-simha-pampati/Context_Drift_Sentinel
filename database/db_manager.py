"""
SQLite Database Manager for Context Drift Sentinel.
Provides thread-safe session storage, message retrieval, and analytical aggregation.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "conversations.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseManager:
    """Manages SQLite database connections, schema migrations, and queries."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a SQLite connection configured for row access."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database schema from schema.sql."""
        schema_file = SCHEMA_PATH
        if not schema_file.exists():
            return
        with open(schema_file, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)

    def save_session(
        self,
        session_id: str,
        session_name: str,
        intent_summary: str,
        messages: List[Dict[str, Any]],
        drift_avg: float = 0.0,
        max_drift: float = 0.0,
        status: str = "NORMAL",
    ) -> None:
        """Save or overwrite a conversation session and all its messages."""
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            # Delete existing session records if exists (or update)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

            conn.execute(
                """
                INSERT INTO sessions (
                    id, session_name, intent_summary, created_at, updated_at,
                    turn_count, drift_avg, max_drift, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    session_name,
                    intent_summary,
                    now,
                    now,
                    len(messages),
                    drift_avg,
                    max_drift,
                    status,
                ),
            )

            for idx, msg in enumerate(messages):
                conn.execute(
                    """
                    INSERT INTO messages (
                        session_id, turn_index, role, content, timestamp,
                        similarity_score, drift_score, drift_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        idx,
                        msg.get("role", "unknown"),
                        msg.get("content", ""),
                        msg.get("timestamp", now),
                        float(msg.get("similarity_score", 1.0)),
                        float(msg.get("drift_score", 0.0)),
                        msg.get("drift_status", "NORMAL"),
                    ),
                )

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve all sessions ordered by creation date descending."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, session_name, intent_summary, created_at,
                       turn_count, drift_avg, max_drift, status
                FROM sessions
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific session along with its messages."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session_row = cursor.fetchone()
            if not session_row:
                return None

            session = dict(session_row)
            msg_cursor = conn.execute(
                """
                SELECT turn_index, role, content, timestamp,
                       similarity_score, drift_score, drift_status
                FROM messages
                WHERE session_id = ?
                ORDER BY turn_index ASC
                """,
                (session_id,),
            )
            session["messages"] = [dict(r) for r in msg_cursor.fetchall()]
            return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and associated messages."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Compute aggregate system metrics across all stored sessions."""
        with self.get_connection() as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            if total_sessions == 0:
                return {
                    "total_sessions": 0,
                    "avg_drift_overall": 0.0,
                    "max_drift_overall": 0.0,
                    "avg_conversation_length": 0.0,
                    "most_stable_session": None,
                    "most_drifted_session": None,
                }

            avg_drift = conn.execute("SELECT AVG(drift_avg) FROM sessions").fetchone()[0] or 0.0
            max_drift = conn.execute("SELECT MAX(max_drift) FROM sessions").fetchone()[0] or 0.0
            avg_len = conn.execute("SELECT AVG(turn_count) FROM sessions").fetchone()[0] or 0.0

            most_stable = conn.execute(
                "SELECT id, session_name, drift_avg, turn_count FROM sessions ORDER BY drift_avg ASC LIMIT 1"
            ).fetchone()
            most_drifted = conn.execute(
                "SELECT id, session_name, max_drift, drift_avg, turn_count FROM sessions ORDER BY max_drift DESC LIMIT 1"
            ).fetchone()

            return {
                "total_sessions": total_sessions,
                "avg_drift_overall": round(float(avg_drift), 2),
                "max_drift_overall": round(float(max_drift), 2),
                "avg_conversation_length": round(float(avg_len), 1),
                "most_stable_session": dict(most_stable) if most_stable else None,
                "most_drifted_session": dict(most_drifted) if most_drifted else None,
            }


_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: Optional[Path | str] = None) -> DatabaseManager:
    """Singleton helper to obtain database manager instance."""
    global _db_instance
    if _db_instance is None or db_path is not None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
