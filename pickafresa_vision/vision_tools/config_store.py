"""
Simple JSON-backed config store for persisting interactive selections across runs.

Config resolution order:
- User-specific file at ~/.pickafresa_vision/config.json
- Project-local file at pickafresa_vision/.user_config.json

Namespacing:
- Use a top-level key per tool/module, e.g., "bbox_depth_auto_pnp_calc", "pnp_calc_manual", "fruit_pose_fusion".

API:
- load_config() -> dict
- save_config(cfg: dict) -> None
- get_namespace(cfg: dict, ns: str) -> dict
- update_namespace(cfg: dict, ns: str, patch: dict) -> None

@aldrick-t, 2025
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_LOCAL = REPO_ROOT / "pickafresa_vision" / ".user_config.json"
USER_DIR = Path.home() / ".pickafresa_vision"
USER_FILE = USER_DIR / "config.json"


def load_config() -> Dict:
    cfg: Dict = {}
    # Load project-local first
    if PROJECT_LOCAL.exists():
        try:
            cfg.update(json.loads(PROJECT_LOCAL.read_text(encoding="utf-8")))
        except Exception:
            pass
    # Override with user-level
    if USER_FILE.exists():
        try:
            user_cfg = json.loads(USER_FILE.read_text(encoding="utf-8"))
            # Deep-merge namespaces
            for k, v in user_cfg.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(cfg: Dict) -> None:
    # Persist to both locations to keep portability
    try:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        USER_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        PROJECT_LOCAL.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_namespace(cfg: Dict, ns: str) -> Dict:
    node = cfg.get(ns)
    if not isinstance(node, dict):
        node = {}
        cfg[ns] = node
    return node


def update_namespace(cfg: Dict, ns: str, patch: Dict) -> None:
    node = get_namespace(cfg, ns)
    node.update(patch)
    save_config(cfg)
