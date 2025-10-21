"""
Fuse fruit camera-frame detections with robot/camera extrinsics to compute
poses in world or robot base frames.

Inputs
------
- Stream of fruit camera-frame transforms from bbox_depth_auto_pnp_calc.py
  (JSON-lines, one object per frame), containing:
    {
      "frame_index": int,
      "fruits": [
        { "index": int, "label": str, "depth_m": float,
          "pos_cam": [x, y, z], "T_cam_fruit": [[...4x4...]] }, ...
      ]
    }
- Fixed extrinsics:
  - T_base_cam (robot base -> camera) OR its inverse T_cam_base
  - Optionally T_world_base if a world frame is desired

Outputs
-------
- For each fruit, print T_world_fruit or T_base_fruit (4x4) and save JSON-lines
  if requested.

Usage (interactive if args omitted):
  python fruit_pose_fusion.py --stream fruits_stream.jsonl --T-base-cam base_cam.json --T-world-base world_base.json

JSON transform format:
  {"T": [[...4x4...]]}

Notes
-----
- If you have the camera pose from PnP (T_world_cam), you can set T_world_base
  by composing with known T_base_cam: T_world_base = T_world_cam @ T_cam_base.
- If the camera is fixed to the robot TCP with a known transform, supply T_base_cam
  constructed from the robot kinematics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence
import numpy as np

# Ensure repository root on sys.path for absolute imports when run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_vision.vision_tools.config_store import load_config, get_namespace, update_namespace


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse fruit camera poses to world or base frames.")
    p.add_argument("--stream", help="Path to JSON-lines stream exported by bbox_depth_auto_pnp_calc.py")
    p.add_argument("--T-base-cam", dest="T_base_cam", help="JSON file with 4x4 T_base_cam (base->camera)")
    p.add_argument("--T-world-base", dest="T_world_base", help="JSON file with 4x4 T_world_base (world->base)")
    p.add_argument("--export", help="Optional JSON-lines output path for fused poses")
    return p.parse_args(argv)


def _prompt_or_default(ns: Dict, key: str, prompt: str, default: str) -> str:
    val = ns.get(key, default)
    ans = input(f"{prompt} [{val}]: ").strip() or str(val)
    ns[key] = ans
    return ans


def _load_T(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "T" in data:
        arr = np.array(data["T"], dtype=float)
    else:
        arr = np.array(data, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Transform at {path} must be 4x4")
    return arr


def _save_line(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


def run(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config()
    ns = get_namespace(cfg, "fruit_pose_fusion")

    stream_path = Path(args.stream) if args.stream else Path(_prompt_or_default(ns, "stream", "Path to fruits JSONL stream", ns.get("stream", "fruits_stream.jsonl")))
    while not stream_path.exists():
        print(f"Stream not found: {stream_path}")
        stream_path = Path(_prompt_or_default(ns, "stream", "Path to fruits JSONL stream", str(stream_path)))

    Tbc_path = Path(args.T_base_cam) if args.T_base_cam else Path(_prompt_or_default(ns, "T_base_cam", "Path to T_base_cam JSON", ns.get("T_base_cam", "T_base_cam.json")))
    Twb_path = Path(args.T_world_base) if args.T_world_base else Path(_prompt_or_default(ns, "T_world_base", "Path to T_world_base JSON (optional)", ns.get("T_world_base", "")))
    export_path = Path(args.export) if args.export else Path(_prompt_or_default(ns, "export", "Export fused poses to JSONL (optional)", ns.get("export", ""))) if ns.get("export") is not None else Path("")

    # Persist user selections
    update_namespace(cfg, "fruit_pose_fusion", {
        "stream": str(stream_path),
        "T_base_cam": str(Tbc_path),
        "T_world_base": str(Twb_path) if Twb_path else "",
        "export": str(export_path) if export_path else "",
    })

    T_base_cam = _load_T(Tbc_path)
    T_world_base = _load_T(Twb_path) if Twb_path and Twb_path.exists() else None

    for line in stream_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        frame_idx = int(obj.get("frame_index", -1))
        fruits = obj.get("fruits", [])
        for f in fruits:
            T_cam_fruit = np.array(f.get("T_cam_fruit"), dtype=float).reshape(4, 4)
            # Compose to base and optionally world
            T_base_fruit = T_base_cam @ T_cam_fruit
            if T_world_base is not None:
                T_world_fruit = T_world_base @ T_base_fruit
            else:
                T_world_fruit = None
            # Print concise output
            pos_b = T_base_fruit[:3, 3]
            print(f"frame={frame_idx} label={f.get('label')} index={f.get('index')} base_pos={pos_b.tolist()}\nT_base_fruit=\n{T_base_fruit}")
            if T_world_fruit is not None:
                pos_w = T_world_fruit[:3, 3]
                print(f"world_pos={pos_w.tolist()}\nT_world_fruit=\n{T_world_fruit}")
            # Export if requested
            if export_path and str(export_path) != "":
                out = {
                    "frame_index": frame_idx,
                    "label": f.get("label"),
                    "index": f.get("index"),
                    "T_base_fruit": T_base_fruit.tolist(),
                }
                if T_world_fruit is not None:
                    out["T_world_fruit"] = T_world_fruit.tolist()
                _save_line(export_path, out)


if __name__ == "__main__":
    run()
