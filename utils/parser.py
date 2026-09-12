"""
Multi-format conversation transcript parser for Context Drift Sentinel.
Supports CSV, JSON, and raw conversational transcripts.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


ROLE_MAPPINGS = {
    "user": "user",
    "human": "user",
    "client": "user",
    "customer": "user",
    "prompter": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "llm": "assistant",
    "model": "assistant",
    "system": "system",
}


def normalize_role(raw_role: str) -> str:
    """Normalize varying role representations to 'user', 'assistant', or 'system'."""
    clean = str(raw_role).strip().lower()
    return ROLE_MAPPINGS.get(clean, clean if clean else "user")


def parse_csv_data(file_content: Union[str, bytes]) -> List[Dict[str, Any]]:
    """Parse CSV conversation data into standard message dictionary list."""
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8", errors="replace")

    df = pd.read_csv(io.StringIO(file_content))
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map content column
    content_col = next((c for c in ["content", "message", "text", "prompt", "utterance"] if c in df.columns), None)
    if not content_col:
        # If no known column, fallback to the longest text column or last column
        content_col = df.columns[-1]

    # Map role column
    role_col = next((c for c in ["role", "speaker", "sender", "author", "from"] if c in df.columns), None)
    
    # Map timestamp column
    time_col = next((c for c in ["timestamp", "time", "date", "created_at"] if c in df.columns), None)

    messages: List[Dict[str, Any]] = []
    now_iso = datetime.utcnow().isoformat()

    for idx, row in df.iterrows():
        content = str(row[content_col]) if pd.notna(row[content_col]) else ""
        if not content.strip():
            continue

        raw_role = str(row[role_col]) if role_col and pd.notna(row[role_col]) else ("user" if idx % 2 == 0 else "assistant")
        role = normalize_role(raw_role)
        timestamp = str(row[time_col]) if time_col and pd.notna(row[time_col]) else now_iso

        messages.append({
            "turn_index": len(messages),
            "role": role,
            "content": content.strip(),
            "timestamp": timestamp,
        })

    return messages


def parse_json_data(file_content: Union[str, bytes]) -> List[Dict[str, Any]]:
    """Parse JSON conversation payload into standard message dictionary list."""
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8", errors="replace")

    data = json.loads(file_content)
    raw_list: List[Dict[str, Any]] = []

    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            raw_list = data["messages"]
        elif "conversation" in data and isinstance(data["conversation"], list):
            raw_list = data["conversation"]
        elif "turns" in data and isinstance(data["turns"], list):
            raw_list = data["turns"]
        else:
            raw_list = [data]

    messages: List[Dict[str, Any]] = []
    now_iso = datetime.utcnow().isoformat()

    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("message") or item.get("text") or item.get("prompt") or ""
        if not str(content).strip():
            continue

        raw_role = item.get("role") or item.get("speaker") or item.get("sender") or ("user" if idx % 2 == 0 else "assistant")
        role = normalize_role(str(raw_role))
        timestamp = str(item.get("timestamp") or item.get("time") or now_iso)

        messages.append({
            "turn_index": len(messages),
            "role": role,
            "content": str(content).strip(),
            "timestamp": timestamp,
        })

    return messages


def parse_raw_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse raw conversational transcript with speaker headers.
    Examples:
        User: How do I implement JWT?
        Assistant: You can use pyjwt...
        [2024-05-01 10:00:00] Human: How about OAuth2?
    """
    lines = raw_text.strip().splitlines()
    messages: List[Dict[str, Any]] = []
    now_iso = datetime.utcnow().isoformat()

    pattern = re.compile(
        r"^(?:\[(?P<timestamp>[^\]]+)\]\s*)?(?P<role>[A-Za-z0-9_\-\s]{2,20})\s*:\s*(?P<content>.*)$",
        re.IGNORECASE,
    )

    current_role = "user"
    current_timestamp = now_iso
    current_content: List[str] = []

    def flush_message():
        if current_content:
            full_text = "\n".join(current_content).strip()
            if full_text:
                messages.append({
                    "turn_index": len(messages),
                    "role": normalize_role(current_role),
                    "content": full_text,
                    "timestamp": current_timestamp,
                })
            current_content.clear()

    for line in lines:
        match = pattern.match(line)
        if match:
            flush_message()
            matched_role = match.group("role").strip()
            matched_time = match.group("timestamp")
            first_line = match.group("content").strip()

            current_role = matched_role
            current_timestamp = matched_time.strip() if matched_time else now_iso
            if first_line:
                current_content.append(first_line)
        else:
            if line.strip():
                current_content.append(line)

    flush_message()

    # Fallback if no speaker headers were parsed
    if not messages and raw_text.strip():
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            messages.append({
                "turn_index": idx,
                "role": "user" if idx % 2 == 0 else "assistant",
                "content": p,
                "timestamp": now_iso,
            })

    return messages


def parse_conversation_file(
    file_bytes_or_str: Union[str, bytes],
    filename_or_format: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Detect format and parse conversation into list of normalized turns.

    Returns:
        (messages, error_message)
    """
    name = filename_or_format.lower()
    try:
        if name.endswith(".csv"):
            return parse_csv_data(file_bytes_or_str), None
        elif name.endswith(".json"):
            return parse_json_data(file_bytes_or_str), None
        else:
            text = file_bytes_or_str if isinstance(file_bytes_or_str, str) else file_bytes_or_str.decode("utf-8", errors="replace")
            # Try JSON parsing first
            try:
                json_res = parse_json_data(text)
                if json_res:
                    return json_res, None
            except Exception:
                pass
            return parse_raw_text(text), None
    except Exception as e:
        return [], f"Failed to parse conversation format: {str(e)}"
