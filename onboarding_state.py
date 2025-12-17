from __future__ import annotations

import json
from pathlib import Path

from paths import user_config_path

_STATE_FILE_NAME = "onboarding_state.json"
_VALID_KEYS = ("menu_done", "demo_done")


class OnboardingState:
    """Store onboarding completion flags in the user's config directory."""

    def __init__(self, path: Path | None = None) -> None:
        base_dir = user_config_path().parent
        base_dir.mkdir(parents=True, exist_ok=True)
        self._path = path or base_dir / _STATE_FILE_NAME
        self._state = {key: False for key in _VALID_KEYS}
        self._load()

    def _load(self) -> None:
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        for key in _VALID_KEYS:
            self._state[key] = bool(data.get(key, False))

    def is_done(self, key: str) -> bool:
        return bool(self._state.get(key, False))

    @property
    def menu_done(self) -> bool:
        return self.is_done("menu_done")

    @property
    def demo_done(self) -> bool:
        return self.is_done("demo_done")

    def mark_done(self, key: str) -> bool:
        if key not in self._state:
            return False
        if self._state[key]:
            return False
        self._state[key] = True
        self._write()
        return True

    def _write(self) -> None:
        suffix = self._path.suffix
        tmp_suffix = f"{suffix}.tmp" if suffix else ".tmp"
        tmp_path = self._path.with_suffix(tmp_suffix)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2, separators=(",", ": "))
        tmp_path.replace(self._path)
