"""
Unit and Integration Tests for Context Drift Sentinel.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from database.db_manager import DatabaseManager
from services.conversation_analyzer import ConversationAnalyzer
from services.drift_detector import DriftDetector
from services.recovery_engine import RecoveryEngine
from utils.parser import parse_conversation_file, parse_csv_data, parse_json_data, parse_raw_text
from utils.similarity import (
    calculate_cosine_similarity,
    calculate_drift_score,
    classify_drift_status,
    compute_exponential_moving_average,
)


class TestSimilarityAndScoring(unittest.TestCase):
    """Test vector math and drift score conversions."""

    def test_cosine_similarity_identical(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([[1.0, 0.0, 0.0]])
        sim = calculate_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim[0], 1.0, places=4)

    def test_cosine_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([[0.0, 1.0]])
        sim = calculate_cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim[0], 0.0, places=4)

    def test_calculate_drift_score_bounds(self):
        self.assertEqual(calculate_drift_score(1.0), 0.0)
        self.assertEqual(calculate_drift_score(0.0), 100.0)
        self.assertEqual(calculate_drift_score(0.5), 50.0)
        self.assertEqual(calculate_drift_score(-0.5), 100.0)  # Capped at 100

    def test_classify_drift_status(self):
        self.assertEqual(classify_drift_status(0.85, 0.65, 0.45), "NORMAL")
        self.assertEqual(classify_drift_status(0.65, 0.65, 0.45), "NORMAL")
        self.assertEqual(classify_drift_status(0.55, 0.65, 0.45), "WARNING")
        self.assertEqual(classify_drift_status(0.40, 0.65, 0.45), "CRITICAL")

    def test_exponential_moving_average(self):
        data = [10.0, 20.0, 30.0]
        ema = compute_exponential_moving_average(data, alpha=0.5)
        self.assertEqual(len(ema), 3)
        self.assertEqual(ema[0], 10.0)
        self.assertEqual(ema[1], 15.0)


class TestParsers(unittest.TestCase):
    """Test conversation format parsing for CSV, JSON, and text transcripts."""

    def test_parse_csv(self):
        csv_text = "role,content\nuser,Hello\nassistant,Hi there!"
        msgs = parse_csv_data(csv_text)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "Hello")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"], "Hi there!")

    def test_parse_json_list(self):
        json_text = '[{"role": "user", "content": "Query 1"}, {"role": "assistant", "content": "Ans 1"}]'
        msgs = parse_json_data(json_text)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["content"], "Ans 1")

    def test_parse_json_nested(self):
        json_text = '{"messages": [{"role": "human", "text": "Nested prompt"}]}'
        msgs = parse_json_data(json_text)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "Nested prompt")

    def test_parse_raw_text(self):
        raw_text = """
        User: How do I test this?
        Assistant: Use pytest.
        User: Thanks!
        """
        msgs = parse_raw_text(raw_text)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "How do I test this?")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[2]["content"], "Thanks!")

    def test_parse_file_dispatcher(self):
        csv_bytes = b"speaker,message\nuser,Test"
        msgs, err = parse_conversation_file(csv_bytes, "test.csv")
        self.assertIsNone(err)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "Test")


class TestDatabaseManager(unittest.TestCase):
    """Test SQLite database operations and dashboard aggregations."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_save_and_retrieve_session(self):
        messages = [
            {"role": "user", "content": "Turn 0", "similarity_score": 1.0, "drift_score": 0.0, "drift_status": "NORMAL"},
            {"role": "assistant", "content": "Turn 1", "similarity_score": 0.8, "drift_score": 20.0, "drift_status": "NORMAL"},
        ]
        self.db.save_session(
            session_id="test_sess_1",
            session_name="Test Session 1",
            intent_summary="Turn 0 intent",
            messages=messages,
            drift_avg=10.0,
            max_drift=20.0,
            status="NORMAL",
        )

        all_sess = self.db.get_all_sessions()
        self.assertEqual(len(all_sess), 1)
        self.assertEqual(all_sess[0]["id"], "test_sess_1")

        loaded = self.db.get_session_by_id("test_sess_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded["messages"]), 2)
        self.assertEqual(loaded["messages"][0]["content"], "Turn 0")

    def test_delete_session(self):
        self.db.save_session("test_del", "Del", "Sum", [{"role": "user", "content": "msg"}])
        self.assertTrue(self.db.delete_session("test_del"))
        self.assertIsNone(self.db.get_session_by_id("test_del"))

    def test_dashboard_metrics(self):
        self.db.save_session("s1", "Session 1", "sum1", [{"role": "user", "content": "a"}], drift_avg=10.0, max_drift=15.0)
        self.db.save_session("s2", "Session 2", "sum2", [{"role": "user", "content": "b"}], drift_avg=50.0, max_drift=80.0)

        metrics = self.db.get_dashboard_metrics()
        self.assertEqual(metrics["total_sessions"], 2)
        self.assertEqual(metrics["avg_drift_overall"], 30.0)
        self.assertEqual(metrics["max_drift_overall"], 80.0)
        self.assertEqual(metrics["most_stable_session"]["id"], "s1")
        self.assertEqual(metrics["most_drifted_session"]["id"], "s2")


class TestRecoveryEngine(unittest.TestCase):
    """Test recovery prompt generation and topic extraction."""

    def setUp(self):
        self.engine = RecoveryEngine()

    def test_topic_extraction(self):
        text = "How do I implement secure JWT authentication with refresh tokens in FastAPI?"
        topic = self.engine.extract_key_topics(text)
        self.assertIsInstance(topic, str)
        self.assertTrue(len(topic) > 0)

    def test_recovery_plan_generation(self):
        ref = "How to build JWT authentication?"
        messages = [
            {"turn_index": 0, "role": "user", "content": "How to build JWT authentication?", "drift_status": "NORMAL"},
            {"turn_index": 1, "role": "assistant", "content": "Use pyjwt.", "drift_status": "NORMAL"},
            {"turn_index": 2, "role": "user", "content": "What is the best recipe for Neapolitan pizza?", "drift_status": "CRITICAL"},
        ]
        plan = self.engine.generate_recovery_plan(ref, messages, inflection_turn=2)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.drift_level, "CRITICAL")
        self.assertEqual(plan.inflection_turn, 2)
        self.assertIn("refocus", plan.suggested_user_prompt.lower())
        self.assertIn("SYSTEM STEERING", plan.system_steering_prompt)


class TestConversationAnalyzer(unittest.TestCase):
    """Test summary analytics computations."""

    def test_compute_session_summary(self):
        messages = [
            {"role": "user", "content": "A", "similarity_score": 1.0, "drift_score": 0.0, "drift_status": "NORMAL"},
            {"role": "assistant", "content": "B", "similarity_score": 0.7, "drift_score": 30.0, "drift_status": "NORMAL"},
            {"role": "user", "content": "C", "similarity_score": 0.4, "drift_score": 60.0, "drift_status": "CRITICAL"},
        ]
        summary = ConversationAnalyzer.compute_session_summary(messages)
        self.assertEqual(summary["total_turns"], 3)
        self.assertEqual(summary["user_turns"], 2)
        self.assertEqual(summary["assistant_turns"], 1)
        self.assertEqual(summary["normal_turns"], 2)
        self.assertEqual(summary["critical_turns"], 1)
        self.assertAlmostEqual(summary["avg_drift"], 30.0, places=1)
        self.assertAlmostEqual(summary["max_drift"], 60.0, places=1)


class TestDriftDetectorMocked(unittest.TestCase):
    """Test DriftDetector with mocked embeddings for fast unit tests."""

    def test_analyze_with_mock_embeddings(self):
        mock_mgr = MagicMock()
        # Mock 3 texts returning 3 vectors
        # Vector 0: ref, Vector 1: similar, Vector 2: orthogonal
        mock_mgr.encode.return_value = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])

        detector = DriftDetector(embedding_manager=mock_mgr)
        messages = [
            {"role": "user", "content": "Initial Intent"},
            {"role": "assistant", "content": "Off topic reply"},
        ]

        result = detector.analyze(messages)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.messages[0]["turn_index"], 0)
        self.assertEqual(result.messages[1]["turn_index"], 1)
        self.assertTrue(result.max_drift > 0)


if __name__ == "__main__":
    unittest.main()
