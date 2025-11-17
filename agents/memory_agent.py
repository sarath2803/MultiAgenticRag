# agents/memory_agent.py
import time
from typing import List, Optional

class MemoryAgent:
    """
    Simple conversational memory agent.
    - Short-term memory: stores last N turns (user+assistant).
    - Long-term memory: optional summary string (manually or auto-updated).
    - Always returns a STRING for prompt injection.
    """

    def __init__(self, max_turns: int = 8):
        # store items as dicts: {"role": "user"/"assistant", "text": "...", "ts": ...}
        self.max_turns = max_turns
        self.history: List[dict] = []
        self.long_term_summary: str = ""

    # --- storage API ---
    def store(self, user_text: Optional[str], assistant_text: Optional[str]) -> None:
        """Store a full turn (user + assistant). Either can be None."""
        if user_text and user_text.strip():
            self.history.append({"role": "user", "text": user_text.strip(), "ts": time.time()})
        if assistant_text and assistant_text.strip():
            self.history.append({"role": "assistant", "text": assistant_text.strip(), "ts": time.time()})
        self._trim_history()

    def add_user(self, text: str) -> None:
        if text and text.strip():
            self.history.append({"role": "user", "text": text.strip(), "ts": time.time()})
            self._trim_history()

    def add_assistant(self, text: str) -> None:
        if text and text.strip():
            self.history.append({"role": "assistant", "text": text.strip(), "ts": time.time()})
            self._trim_history()

    def _trim_history(self):
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    # --- retrieval API ---
    def get_recent_items(self, n: int = 6) -> List[dict]:
        """Return last n history items (most recent)."""
        return self.history[-n:]

    def get_context_text(self, n: int = 6) -> str:
        """Return a nicely formatted string for prompt injection (last n turns + long-term summary)."""
        parts = []
        if self.long_term_summary:
            parts.append("[Long-term summary]")
            parts.append(self.long_term_summary)
            parts.append("")  # blank line

        recent = self.get_recent_items(n)
        if recent:
            parts.append("[Recent conversation]")
            for it in recent:
                role = it.get("role", "").capitalize()
                parts.append(f"{role}: {it.get('text','')}")
        return "\n".join(parts).strip()

    def recall(self, n: int = 6) -> str:
        """Alias for get_context_text."""
        return self.get_context_text(n)

    def get_relevant_memory(self, query: str, n: int = 6) -> str:
        """
        Simple relevance: for now return recent items as a string.
        (This is the place to implement embedding-based memory search later.)
        """
        return self.get_context_text(n)

    # --- long-term summary API ---
    def update_long_term_summary(self, summary_text: str) -> None:
        if summary_text and isinstance(summary_text, str):
            self.long_term_summary = summary_text.strip()

    def clear(self) -> None:
        self.history = []
        self.long_term_summary = ""
