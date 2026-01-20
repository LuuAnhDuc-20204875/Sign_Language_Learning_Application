from __future__ import annotations

import json
import os
import re
import secrets
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


def _pbkdf2_hash(password: str, salt: bytes, iterations: int = 150_000) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return dk.hex()


@dataclass
class AuthResult:
    ok: bool
    message: str = ""
    username: Optional[str] = None


class UserStore:
    """
    Local-only user auth + high score storage.
    File format (users.json):
    {
      "users": {
        "alice": {
          "salt": "...hex...",
          "hash": "...hex...",
          "iterations": 150000,
          "high_scores": {"snake": 10, "quiz": 5, "spelling": 3}
        }
      },
      "guest": { "high_scores": {...} }
    }
    """

    def __init__(self, path: str = "users.json"):
        self.path = path
        self.data: Dict = {"users": {}, "guest": {"high_scores": {}}}
        self.load()

    # ---------- IO ----------
    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                # never crash app because of storage
                self.data = {"users": {}, "guest": {"high_scores": {}}}
        else:
            self.data = {"users": {}, "guest": {"high_scores": {}}}

        # normalize
        self.data.setdefault("users", {})
        self.data.setdefault("guest", {"high_scores": {}})
        self.data["guest"].setdefault("high_scores", {})

    def _atomic_save(self, payload: Dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def save(self) -> None:
        try:
            self._atomic_save(self.data)
        except Exception:
            pass

    # ---------- Auth ----------
    def validate_username(self, username: str) -> AuthResult:
        u = (username or "").strip()
        if not u:
            return AuthResult(False, "Vui lòng nhập tên đăng nhập.")
        if not _USERNAME_RE.match(u):
            return AuthResult(False, "Tên đăng nhập chỉ gồm chữ/số/_ và dài 3–20 ký tự.")
        return AuthResult(True, username=u, message="OK")

    def validate_password(self, password: str) -> AuthResult:
        p = password or ""
        if len(p) < 4:
            return AuthResult(False, "Mật khẩu tối thiểu 4 ký tự.")
        return AuthResult(True, message="OK")

    def register(self, username: str, password: str) -> AuthResult:
        v1 = self.validate_username(username)
        if not v1.ok:
            return v1
        v2 = self.validate_password(password)
        if not v2.ok:
            return v2

        u = v1.username
        if u in self.data.get("users", {}):
            return AuthResult(False, "Tên đăng nhập đã tồn tại.")

        salt = secrets.token_bytes(16)
        iterations = 150_000
        h = _pbkdf2_hash(password, salt, iterations)

        self.data["users"][u] = {
            "salt": salt.hex(),
            "hash": h,
            "iterations": iterations,
            "high_scores": {"snake": 0, "quiz": 0, "spelling": 0, "mcq": 0},
        }
        self.save()
        return AuthResult(True, "Đăng ký thành công.", username=u)

    def login(self, username: str, password: str) -> AuthResult:
        u = (username or "").strip()
        if u not in self.data.get("users", {}):
            return AuthResult(False, "Sai tên đăng nhập hoặc mật khẩu.")
        user = self.data["users"][u]
        try:
            salt = bytes.fromhex(user.get("salt", ""))
            iterations = int(user.get("iterations", 150_000))
        except Exception:
            return AuthResult(False, "Tài khoản bị lỗi dữ liệu (salt/iterations).")

        h = _pbkdf2_hash(password or "", salt, iterations)
        if h != user.get("hash", ""):
            return AuthResult(False, "Sai tên đăng nhập hoặc mật khẩu.")
        return AuthResult(True, "Đăng nhập thành công.", username=u)

    # ---------- Scores ----------
    def _user_bucket(self, username: Optional[str]) -> Dict:
        if username:
            return self.data["users"].setdefault(username, {"high_scores": {}})
        return self.data.setdefault("guest", {"high_scores": {}})

    def get_high_score(self, username: Optional[str], mode: str) -> int:
        bucket = self._user_bucket(username)
        hs = bucket.setdefault("high_scores", {})
        try:
            return int(hs.get(mode, 0))
        except Exception:
            return 0

    def set_high_score(self, username: Optional[str], mode: str, score: int) -> int:
        bucket = self._user_bucket(username)
        hs = bucket.setdefault("high_scores", {})
        current = int(hs.get(mode, 0) or 0)
        score = int(score or 0)
        if score > current:
            hs[mode] = score
            self.save()
            return score
        return current
