"""
Centralized configuration for pickafresa_vision.

Loads environment variables (optionally from a local .env file) and provides
helpers to access secrets and configuration with clear errors if missing.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    # Load environment from a local .env if present (no-op if missing)
    from dotenv import load_dotenv  # type: ignore

    # Attempt to load a .env placed either at repo root or within pickafresa_vision/
    # Priority: local folder .env > repo root .env
    local_env = Path(__file__).with_name(".env")
    root_env = Path(__file__).resolve().parents[1] / ".env"
    for env_path in (local_env, root_env):
        if env_path.exists():
            load_dotenv(env_path)  # Load without overriding system env
            break
except Exception:
    # If python-dotenv is not installed, we silently continue; OS env will still work
    pass


class ConfigError(RuntimeError):
    pass


def require_env(name: str) -> str:
    """Return the env var value or raise a helpful error if missing/empty."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        hint = (
            f"Missing required environment variable '{name}'. "
            "Create pickafresa_vision/.env from .env.example, set the value, "
            "or export it in your shell/CI environment."
        )
        raise ConfigError(hint)
    return val


def get_env(name: str, default: str | None = None) -> str | None:
    """Return the env var value or the provided default without raising."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val
