from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Stat:
    total: int = 0
    correct: int = 0

    @property
    def acc(self) -> float:
        return (self.correct / self.total) if self.total > 0 else 0.0


class ProgressManager:
    """Safe JSON progress logger.
    Structure:
    {
      "letters": { "A": {"total": 10, "correct": 8}, ... },
      "words":   { "HELLO": {"total": 3, "correct": 2}, ... }
    }
    """

    def __init__(self, path: str = "progress.json"):
        self.path = path
        self.data: Dict[str, Dict[str, Dict[str, int]]] = {"letters": {}, "words": {}}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {"letters": {}, "words": {}}
        else:
            self.data = {"letters": {}, "words": {}}

    def _atomic_save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def save(self) -> None:
        try:
            self._atomic_save()
        except Exception:
            # Never crash the app because of logging.
            pass

    def record_letter(self, letter: str | None, correct: bool) -> None:
        if not letter:
            return
        letters = self.data.setdefault("letters", {})
        stat = letters.setdefault(letter, {"total": 0, "correct": 0})
        stat["total"] += 1
        if correct:
            stat["correct"] += 1
        self.save()

    def record_word(self, word: str | None, correct: bool) -> None:
        if not word:
            return
        words = self.data.setdefault("words", {})
        stat = words.setdefault(word, {"total": 0, "correct": 0})
        stat["total"] += 1
        if correct:
            stat["correct"] += 1
        self.save()

    def get_letter_stat(self, letter: str) -> Stat:
        lt = (letter or "").strip().upper()
        v = self.data.get("letters", {}).get(lt, {"total": 0, "correct": 0})
        return Stat(total=int(v.get("total", 0) or 0), correct=int(v.get("correct", 0) or 0))

    def suggestion_candidate(self, min_attempts: int = 6, acc_threshold: float = 0.70) -> Tuple[Optional[str], float, int]:
        """Return (letter, acc, total) for the weakest letter that meets criteria."""
        weak = self.get_weak_letters(top_k=1)
        if not weak:
            return None, 0.0, 0
        lt, acc, total = weak[0]
        if int(total) < int(min_attempts):
            return None, float(acc), int(total)
        if float(acc) >= float(acc_threshold):
            return None, float(acc), int(total)
        return lt, float(acc), int(total)

    @staticmethod
    def _safe_user_key(username: Optional[str]) -> str:
        u = (username or "guest").strip()
        u = re.sub(r"[^A-Za-z0-9_\-]+", "_", u)[:30] or "guest"
        return u

    @classmethod
    def for_user(cls, username: Optional[str], base_dir: str = ".") -> "ProgressManager":
        """Per-user progress file: progress_<user>.json"""
        key = cls._safe_user_key(username)
        path = os.path.join(base_dir, f"progress_{key}.json")
        return cls(path=path)

    def get_weak_letters(self, top_k: int = 3) -> List[Tuple[str, float, int]]:
        letters = self.data.get("letters", {})
        if not letters:
            return []
        stats: List[Tuple[str, float, int]] = []
        for lt, v in letters.items():
            total = max(1, int(v.get("total", 0)))
            correct = int(v.get("correct", 0))
            acc = correct / total
            stats.append((lt, acc, int(v.get("total", 0))))
        stats.sort(key=lambda x: x[1])
        return stats[:top_k]
