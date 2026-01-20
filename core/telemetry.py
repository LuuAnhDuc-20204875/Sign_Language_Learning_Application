from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _safe_user_key(username: Optional[str]) -> str:
    u = (username or "guest").strip()
    u = re.sub(r"[^A-Za-z0-9_\-]+", "_", u)[:30] or "guest"
    return u


def _utc_iso(ts: Optional[float] = None) -> str:
    dt = datetime.utcfromtimestamp(float(ts or time.time()))
    # ISO without microseconds for readability
    return dt.replace(microsecond=0).isoformat() + "Z"


class EventLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: str):
        self.path = path
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception:
            pass

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Write one event line. Never raises."""
        try:
            obj = {
                "ts": _utc_iso(),
                "type": str(event_type),
                **(payload or {}),
            }
            line = json.dumps(obj, ensure_ascii=False)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Never crash the app due to logging.
            return


@dataclass
class _ModeAgg:
    attempts: int = 0
    correct: int = 0
    completions: int = 0
    # Confidence accumulation for gesture recognition modes
    conf_sum: float = 0.0
    # Time accumulation
    time_sum_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": int(self.attempts),
            "correct": int(self.correct),
            "completions": int(self.completions),
            "conf_sum": float(self.conf_sum),
            "time_sum_sec": float(self.time_sum_sec),
        }


class StatsManager:
    """Safe JSON aggregate stats for thesis tables.

    Schema (v1):
    {
      "schema_version": 1,
      "user": "alice",
      "updated_at": "...",
      "modes": {
         "quiz": {"attempts":..., "correct":..., "completions":..., "conf_sum":..., "time_sum_sec":...},
         ...
      },
      "targets": {
         "quiz": {"A": {"attempts":..., "correct":..., "time_sum_sec":...}, ...},
         "mcq":  {"A": {"questions":..., "correct":..., "rt_sum_sec":...}, ...},
         "spelling_word": {"HELLO": {"completed":..., "time_sum_sec":...}, ...}
      }
    }
    """

    def __init__(self, path: str, user_key: str):
        self.path = path
        self.user_key = user_key
        self.data: Dict[str, Any] = {}
        self._load()

    @classmethod
    def for_user(cls, username: Optional[str], base_dir: str = "logs") -> "StatsManager":
        key = _safe_user_key(username)
        path = os.path.join(base_dir, f"stats_{key}.json")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        return cls(path=path, user_key=key)

    def _default(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "user": self.user_key,
            "updated_at": _utc_iso(),
            "modes": {},
            "targets": {},
        }

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = self._default()
        else:
            self.data = self._default()
        self.data.setdefault("schema_version", 1)
        self.data.setdefault("user", self.user_key)
        self.data.setdefault("modes", {})
        self.data.setdefault("targets", {})

    def _atomic_save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def save(self) -> None:
        try:
            self.data["updated_at"] = _utc_iso()
            self._atomic_save()
        except Exception:
            return

    # ---------- Helpers ----------
    def _mode_bucket(self, mode: str) -> Dict[str, Any]:
        m = self.data.setdefault("modes", {})
        b = m.setdefault(mode, _ModeAgg().to_dict())
        # Ensure keys exist
        b.setdefault("attempts", 0)
        b.setdefault("correct", 0)
        b.setdefault("completions", 0)
        b.setdefault("conf_sum", 0.0)
        b.setdefault("time_sum_sec", 0.0)
        return b

    def _target_bucket(self, group: str, key: str, default: Dict[str, Any]) -> Dict[str, Any]:
        t = self.data.setdefault("targets", {})
        g = t.setdefault(group, {})
        b = g.setdefault(key, dict(default))
        # Normalize defaults
        for k, v in default.items():
            b.setdefault(k, v)
        return b

    # ---------- Public API ----------
    def record_gesture_attempt(
        self,
        mode: str,
        target: str,
        correct: bool,
        conf: float = 0.0,
        dt_sec: float = 0.0,
    ) -> None:
        """Attempt = one accepted stable prediction for a target."""
        try:
            mb = self._mode_bucket(mode)
            mb["attempts"] = int(mb.get("attempts", 0)) + 1
            if bool(correct):
                mb["correct"] = int(mb.get("correct", 0)) + 1
            mb["conf_sum"] = float(mb.get("conf_sum", 0.0)) + float(conf or 0.0)
            mb["time_sum_sec"] = float(mb.get("time_sum_sec", 0.0)) + float(dt_sec or 0.0)

            tb = self._target_bucket(
                group=str(mode),
                key=str(target),
                default={"attempts": 0, "correct": 0, "time_sum_sec": 0.0},
            )
            tb["attempts"] = int(tb.get("attempts", 0)) + 1
            if bool(correct):
                tb["correct"] = int(tb.get("correct", 0)) + 1
            tb["time_sum_sec"] = float(tb.get("time_sum_sec", 0.0)) + float(dt_sec or 0.0)
            self.save()
        except Exception:
            return

    def record_gesture_completion(self, mode: str, target: str, time_to_complete_sec: float = 0.0) -> None:
        """Completion = achieved required streak for a target."""
        try:
            mb = self._mode_bucket(mode)
            mb["completions"] = int(mb.get("completions", 0)) + 1
            mb["time_sum_sec"] = float(mb.get("time_sum_sec", 0.0)) + float(time_to_complete_sec or 0.0)

            tb = self._target_bucket(
                group=f"{mode}_completion",
                key=str(target),
                default={"completions": 0, "time_sum_sec": 0.0},
            )
            tb["completions"] = int(tb.get("completions", 0)) + 1
            tb["time_sum_sec"] = float(tb.get("time_sum_sec", 0.0)) + float(time_to_complete_sec or 0.0)
            self.save()
        except Exception:
            return

    def record_mcq_question(self, target: str) -> None:
        try:
            mb = self._mode_bucket("mcq")
            mb["attempts"] = int(mb.get("attempts", 0)) + 1

            tb = self._target_bucket(
                group="mcq",
                key=str(target),
                default={"questions": 0, "correct": 0, "rt_sum_sec": 0.0},
            )
            tb["questions"] = int(tb.get("questions", 0)) + 1
            self.save()
        except Exception:
            return

    def record_mcq_answer(self, target: str, correct: bool, reaction_time_sec: float = 0.0) -> None:
        try:
            mb = self._mode_bucket("mcq")
            if bool(correct):
                mb["correct"] = int(mb.get("correct", 0)) + 1

            tb = self._target_bucket(
                group="mcq",
                key=str(target),
                default={"questions": 0, "correct": 0, "rt_sum_sec": 0.0},
            )
            if bool(correct):
                tb["correct"] = int(tb.get("correct", 0)) + 1
            tb["rt_sum_sec"] = float(tb.get("rt_sum_sec", 0.0)) + float(reaction_time_sec or 0.0)
            self.save()
        except Exception:
            return

    def record_spelling_word_complete(self, word: str, time_sec: float = 0.0) -> None:
        try:
            mb = self._mode_bucket("spelling")
            mb["completions"] = int(mb.get("completions", 0)) + 1
            mb["time_sum_sec"] = float(mb.get("time_sum_sec", 0.0)) + float(time_sec or 0.0)

            tb = self._target_bucket(
                group="spelling_word",
                key=str(word),
                default={"completed": 0, "time_sum_sec": 0.0},
            )
            tb["completed"] = int(tb.get("completed", 0)) + 1
            tb["time_sum_sec"] = float(tb.get("time_sum_sec", 0.0)) + float(time_sec or 0.0)
            self.save()
        except Exception:
            return

    def record_snake_session_start(self) -> None:
        try:
            mb = self._mode_bucket("snake")
            mb["completions"] = int(mb.get("completions", 0)) + 1  # treat as sessions count
            self.save()
        except Exception:
            return

    def record_snake_food_eaten(self) -> None:
        try:
            tb = self._target_bucket(
                group="snake",
                key="food",
                default={"eaten": 0},
            )
            tb["eaten"] = int(tb.get("eaten", 0)) + 1
            self.save()
        except Exception:
            return

    def record_snake_game_over(self, score: int, duration_sec: float = 0.0) -> None:
        try:
            tb = self._target_bucket(
                group="snake",
                key="game_over",
                default={"count": 0, "best_score": 0, "time_sum_sec": 0.0},
            )
            tb["count"] = int(tb.get("count", 0)) + 1
            tb["best_score"] = max(int(tb.get("best_score", 0)), int(score or 0))
            tb["time_sum_sec"] = float(tb.get("time_sum_sec", 0.0)) + float(duration_sec or 0.0)

            mb = self._mode_bucket("snake")
            mb["time_sum_sec"] = float(mb.get("time_sum_sec", 0.0)) + float(duration_sec or 0.0)
            self.save()
        except Exception:
            return


class Telemetry:
    """Convenience wrapper: EventLogger + StatsManager."""

    def __init__(self, events: EventLogger, stats: StatsManager, user_key: str):
        self.events = events
        self.stats = stats
        self.user_key = user_key

    @classmethod
    def for_user(cls, username: Optional[str], base_dir: str = "logs") -> "Telemetry":
        key = _safe_user_key(username)
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            pass
        events_path = os.path.join(base_dir, f"events_{key}.jsonl")
        return cls(events=EventLogger(events_path), stats=StatsManager.for_user(username, base_dir=base_dir), user_key=key)

    # Thin wrappers that also emit event lines
    def mode_enter(self, mode: str) -> None:
        self.events.log("mode_enter", {"user": self.user_key, "mode": str(mode)})

    def gesture_attempt(self, mode: str, target: str, pred: str, correct: bool, conf: float, dt_sec: float) -> None:
        self.stats.record_gesture_attempt(mode=mode, target=target, correct=correct, conf=conf, dt_sec=dt_sec)
        self.events.log(
            "gesture_attempt",
            {
                "user": self.user_key,
                "mode": str(mode),
                "target": str(target),
                "pred": str(pred),
                "correct": bool(correct),
                "conf": float(conf or 0.0),
                "dt_sec": float(dt_sec or 0.0),
            },
        )

    def gesture_completion(self, mode: str, target: str, time_to_complete_sec: float) -> None:
        self.stats.record_gesture_completion(mode=mode, target=target, time_to_complete_sec=time_to_complete_sec)
        self.events.log(
            "gesture_complete",
            {
                "user": self.user_key,
                "mode": str(mode),
                "target": str(target),
                "time_to_complete_sec": float(time_to_complete_sec or 0.0),
            },
        )

    def mcq_question(self, target: str, options: Dict[str, str]) -> None:
        self.stats.record_mcq_question(target=target)
        self.events.log("mcq_question", {"user": self.user_key, "target": str(target), "options": dict(options or {})})

    def mcq_answer(self, target: str, chosen: str, correct: bool, reaction_time_sec: float) -> None:
        self.stats.record_mcq_answer(target=target, correct=correct, reaction_time_sec=reaction_time_sec)
        self.events.log(
            "mcq_answer",
            {
                "user": self.user_key,
                "target": str(target),
                "chosen": str(chosen),
                "correct": bool(correct),
                "reaction_time_sec": float(reaction_time_sec or 0.0),
            },
        )

    def spelling_word_complete(self, word: str, time_sec: float) -> None:
        self.stats.record_spelling_word_complete(word=word, time_sec=time_sec)
        self.events.log("spelling_word_complete", {"user": self.user_key, "word": str(word), "time_sec": float(time_sec or 0.0)})

    def snake_session_start(self) -> None:
        self.stats.record_snake_session_start()
        self.events.log("snake_session_start", {"user": self.user_key})

    def snake_food_eaten(self, score: int) -> None:
        self.stats.record_snake_food_eaten()
        self.events.log("snake_food_eaten", {"user": self.user_key, "score": int(score or 0)})

    def snake_game_over(self, score: int, duration_sec: float) -> None:
        self.stats.record_snake_game_over(score=score, duration_sec=duration_sec)
        self.events.log("snake_game_over", {"user": self.user_key, "score": int(score or 0), "duration_sec": float(duration_sec or 0.0)})
