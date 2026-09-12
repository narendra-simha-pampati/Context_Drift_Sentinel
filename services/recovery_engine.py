"""
Rule-Based and Semantic Recovery Suggestion Engine.
Generates actionable prompt steering interventions without external LLM API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class RecoverySuggestion:
    """Structure holding recovery recommendations and steering prompts."""
    drift_level: str  # "WARNING" | "CRITICAL"
    original_topic: str
    shifted_topic: str
    suggested_user_prompt: str
    system_steering_prompt: str
    context_pruning_recommendation: str
    inflection_turn: Optional[int] = None


class RecoveryEngine:
    """Extracts key themes and generates actionable prompts to steer drifted conversations."""

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
        "to", "for", "with", "about", "as", "by", "that", "this", "it", "from", "be",
        "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
        "will", "how", "what", "why", "when", "where", "which", "who", "i", "you", "we",
        "they", "he", "she", "me", "my", "your", "our", "their", "please", "help",
        "want", "need", "like", "using", "used", "make", "create", "write", "code",
    }

    def extract_key_topics(self, text: str, max_keywords: int = 3) -> str:
        """Extract dominant keywords/phrases from text using TF-IDF and n-gram analysis."""
        clean_text = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = [w for w in clean_text.split() if w not in self.STOP_WORDS and len(w) > 2]
        
        if not tokens:
            return "the original topic"

        if len(tokens) <= 3:
            return " ".join(tokens)

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10,
            )
            tfidf_matrix = vectorizer.fit_transform([clean_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort features by score
            sorted_indices = scores.argsort()[::-1]
            top_terms = [feature_names[i] for i in sorted_indices[:max_keywords] if scores[i] > 0]
            
            if top_terms:
                return ", ".join(top_terms)
        except Exception:
            pass

        # Fallback to frequency
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
        return ", ".join(sorted_tokens[:max_keywords])

    def generate_recovery_plan(
        self,
        reference_intent: str,
        messages: List[Dict[str, Any]],
        inflection_turn: Optional[int] = None,
    ) -> Optional[RecoverySuggestion]:
        """
        Analyze drifted messages and generate targeted recovery prompts.
        """
        drifted_messages = [
            m for m in messages if m.get("drift_status") in ["WARNING", "CRITICAL"]
        ]

        if not drifted_messages:
            return None

        # Determine severity
        critical_count = sum(1 for m in drifted_messages if m.get("drift_status") == "CRITICAL")
        drift_level = "CRITICAL" if critical_count > 0 else "WARNING"

        # Extract original topic
        original_topic = self.extract_key_topics(reference_intent, max_keywords=3)

        # Extract drifted topic from the latest/worst drifted messages
        latest_drifted_text = " ".join([m.get("content", "") for m in drifted_messages[-3:]])
        shifted_topic = self.extract_key_topics(latest_drifted_text, max_keywords=3)

        # If topics overlap identically, refine
        if shifted_topic.lower() == original_topic.lower():
            shifted_topic = "tangential implementation details"

        # Generate User Recovery Prompt
        user_prompt = (
            f"Let's refocus back on our original goal regarding {original_topic}. "
            f"Please set aside discussions on {shifted_topic} and address the primary requirement."
        )

        # Generate System Steering Prompt (for enterprise LLM system injection)
        first_intent_snippet = reference_intent[:120].strip().replace('"', "'")
        system_prompt = (
            f"[SYSTEM STEERING INTERVENTION]\n"
            f"Context Drift Detected: Conversation has diverged towards '{shifted_topic}'.\n"
            f"Anchor Objective: \"{first_intent_snippet}\"\n"
            f"Directive: Immediately steer output back to '{original_topic}'. Do not elaborate on '{shifted_topic}'."
        )

        # Context Pruning suggestion
        first_drift_idx = inflection_turn if inflection_turn is not None else drifted_messages[0].get("turn_index", 1)
        last_turn_idx = messages[-1].get("turn_index", len(messages) - 1)
        pruning_rec = (
            f"Consider pruning conversation history between Turn {first_drift_idx} and Turn {last_turn_idx} "
            f"or summarizing them into a 1-line recap before subsequent user turns to save context window tokens."
        )

        return RecoverySuggestion(
            drift_level=drift_level,
            original_topic=original_topic,
            shifted_topic=shifted_topic,
            suggested_user_prompt=user_prompt,
            system_steering_prompt=system_prompt,
            context_pruning_recommendation=pruning_rec,
            inflection_turn=first_drift_idx,
        )
