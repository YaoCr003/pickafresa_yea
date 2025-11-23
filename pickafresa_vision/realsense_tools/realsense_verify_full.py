'''
RealSense D400 Series Full Stream Verification Tool
Tests combined color + depth stream configurations to ensure bandwidth is sufficient.

Supports three testing modes:
- 'paired': Tests matching resolution/fps for both streams (fastest)
- 'independent': Tests best of each stream independently, then validates together
- 'comprehensive': Tests all combinations of color × depth profiles (slowest, most thorough)

Provides both CLI tool and API for programmatic use.

@aldrick-t, 2025
'''

from __future__ import annotations
import sys
import time
import os
import logging
import json
import platform
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

try:
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    HAVE_REALSENSE = True
    HAVE_OPENCV = True
except ImportError as e:
    HAVE_REALSENSE = False
    HAVE_OPENCV = False
    print(f"Warning: Missing dependencies - {e}")

# Add repo root to path for config_store import
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pickafresa_vision.vision_tools.config_store import (
    load_config,
    save_config,
    get_namespace,
    update_namespace,
)

# Configuration namespace
CONFIG_NAMESPACE = "realsense_verify_full"

# Setup logging
def setup_logger(log_file: Path) -> logging.Logger:
    """Setup structured JSON logger for full verification."""
    logger = logging.getLogger("realsense_full_verify")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "message": record.getMessage(),
            }
            if hasattr(record, 'extra_data'):
                log_data.update(record.extra_data)
            return json.dumps(log_data)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)
    return logger

def log_with_data(logger: logging.Logger, level: int, message: str, **kwargs):
    extra = {'extra_data': kwargs} if kwargs else {}
    logger.log(level, message, extra=extra)

def get_system_info() -> Dict[str, Any]:
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    return info

# Import color and depth verifiers
try:
    from realsense_verify_color import (
        DEFAULT_COLOR_PROFILES,
        get_camera_serial,
        ColorProfileResult,
        get_best_color_profile,
    )
    from realsense_verify_depth import (
        DEFAULT_DEPTH_PROFILES,
        DepthProfileResult,
        get_best_depth_profile,
    )
    HAVE_VERIFIERS = True
except ImportError:
    HAVE_VERIFIERS = False
    print("Warning: Could not import color/depth verifiers")


@dataclass
class FullProfileResult:
    """Result of testing a combined color + depth profile configuration."""
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int
    success: bool
    message: str
    
    @property
    def color_config(self) -> Tuple[int, int, int]:
        """Get color configuration as (width, height, fps) tuple."""
        return (self.color_width, self.color_height, self.color_fps)
    
    @property
    def depth_config(self) -> Tuple[int, int, int]:
        """Get depth configuration as (width, height, fps) tuple."""
        return (self.depth_width, self.depth_height, self.depth_fps)
    
    @property
    def config_tuple(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get full configuration as (color_config, depth_config) tuple."""
        return (self.color_config, self.depth_config)
    
    def __repr__(self) -> str:
        status = "[OK]" if self.success else "[FAIL]"
        return (f"{status} Color:{self.color_width}x{self.color_height}@{self.color_fps}fps "
                f"Depth:{self.depth_width}x{self.depth_height}@{self.depth_fps}fps: {self.message}")


def save_working_profiles(serial: str, profiles: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]):
    """Save working full (color+depth) profiles for a specific camera serial."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    if "camera_profiles" not in ns:
        ns["camera_profiles"] = {}
    
    ns["camera_profiles"][serial] = {
        "profiles": [[list(c), list(d)] for c, d in profiles],
        "last_verified": time.time(),
    }
    
    save_config(cfg)


def load_working_profiles(serial: str) -> Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
    """Load saved working full profiles for a specific camera serial."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    camera_profiles = ns.get("camera_profiles", {})
    if serial not in camera_profiles:
        return None
    
    profiles_data = camera_profiles[serial].get("profiles", [])
    return [(tuple(c), tuple(d)) for c, d in profiles_data]


def verify_full_profile(
    color_config: Tuple[int, int, int],
    depth_config: Tuple[int, int, int],
    verbose: bool = False,
    timeout_ms: int = 2000,
    max_retries: int = 10,
    show_preview: bool = False,
    serial: Optional[str] = None,
) -> FullProfileResult:
    """
    Test a single combined color + depth configuration.
    
    Args:
        color_config: (width, height, fps) for color stream
        depth_config: (width, height, fps) for depth stream
        verbose: Whether to print progress messages
        timeout_ms: Timeout in milliseconds for wait_for_frames
        max_retries: Maximum number of frame read attempts
        show_preview: Whether to show live preview window (for debugging)
        serial: Specific device serial to target (auto-detected if None)
    
    Returns:
        FullProfileResult with test outcome
    """
    if not HAVE_REALSENSE:
        raise RuntimeError("pyrealsense2 is required but not installed")
    
    cw, ch, cfps = color_config
    dw, dh, dfps = depth_config
    
    # Resolve device serial once so librealsense does not probe unintended sensors.
    device_serial = serial
    if device_serial is None:
        try:
            device_serial = get_camera_serial()
        except Exception as e:
            if verbose:
                print(f"Warning: could not query RealSense serial - {e}")
            device_serial = None
    if not device_serial:
        return FullProfileResult(
            cw, ch, cfps, dw, dh, dfps,
            False,
            "No RealSense device detected"
        )
    
    if verbose:
        print(f"\nTesting Color:{cw}x{ch}@{cfps}fps + Depth:{dw}x{dh}@{dfps}fps")

    # Prefer DIRECT sensor mode on macOS to avoid pipeline/device_hub power toggles
    use_pipeline = os.environ.get("REALSENSE_FULL_USE_PIPELINE", "0") == "1"
    log_path = REPO_ROOT / "pickafresa_vision" / "logs" / "realsense_full_verify.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    logger = setup_logger(log_path)

    # Log start and environment
    log_with_data(
        logger, logging.INFO, "Starting full profile verification",
        color_config={"width": cw, "height": ch, "fps": cfps},
        depth_config={"width": dw, "height": dh, "fps": dfps},
        device_serial=device_serial,
        system_info=get_system_info(),
        use_pipeline=use_pipeline,
        platform=sys.platform,
    )

    if sys.platform == "darwin" and not use_pipeline:
        try:
            log_with_data(logger, logging.INFO, "Entering DIRECT full sensor mode")
            ctx2 = rs.context()
            dev2 = None
            for d in ctx2.devices:
                if (not device_serial) or d.get_info(rs.camera_info.serial_number) == device_serial:
                    dev2 = d
                    break
            if dev2 is None:
                return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, "No RealSense device matched for direct mode")

            # Find sensors exposing the required streams
            color_sensor = None
            depth_sensor = None
            for s in dev2.query_sensors():
                # Probe stream profiles once per sensor
                try:
                    s_profiles = s.get_stream_profiles()
                except Exception:
                    s_profiles = []
                if any(p.stream_type() == rs.stream.color for p in s_profiles):
                    color_sensor = color_sensor or s
                if any(p.stream_type() == rs.stream.depth for p in s_profiles):
                    depth_sensor = depth_sensor or s
            if color_sensor is None or depth_sensor is None:
                return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, "Required sensors not found (color/depth)")

            # Select matching profiles
            color_match = None
            try:
                for p in color_sensor.get_stream_profiles():
                    try:
                        if p.stream_type() != rs.stream.color:
                            continue
                        vsp = p.as_video_stream_profile()
                        if not vsp:
                            continue
                        if vsp.width() == cw and vsp.height() == ch and p.fps() == cfps and \
                           p.format() in (rs.format.bgr8, rs.format.rgb8, rs.format.yuyv):
                            color_match = p
                            break
                    except Exception:
                        continue
            except Exception:
                pass

            depth_match = None
            try:
                for p in depth_sensor.get_stream_profiles():
                    try:
                        if p.stream_type() != rs.stream.depth:
                            continue
                        vsp = p.as_video_stream_profile()
                        if not vsp:
                            continue
                        if vsp.width() == dw and vsp.height() == dh and p.fps() == dfps and \
                           p.format() == rs.format.z16:
                            depth_match = p
                            break
                    except Exception:
                        continue
            except Exception:
                pass

            if color_match is None:
                return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, "No matching color profile on sensor")
            if depth_match is None:
                return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, "No matching depth profile on sensor")

            # Start both sensors with queues
            q_color = rs.frame_queue(1)
            q_depth = rs.frame_queue(1)
            color_opened = depth_opened = False
            color_started = depth_started = False
            try:
                log_with_data(
                    logger, logging.DEBUG, "DIRECT open/start",
                    color={"width": cw, "height": ch, "fps": cfps},
                    depth={"width": dw, "height": dh, "fps": dfps},
                )

                time.sleep(0.1)
                color_sensor.open(color_match)
                color_opened = True
                depth_sensor.open(depth_match)
                depth_opened = True
                time.sleep(0.05)
                color_sensor.start(q_color)
                color_started = True
                depth_sensor.start(q_depth)
                depth_started = True

                ok_color = ok_depth = False
                cf = None
                df = None
                first_exc = None
                for _ in range(max_retries):
                    try:
                        if not ok_color:
                            cf = q_color.wait_for_frame(timeout_ms // 2)
                            if cf:
                                ok_color = True
                        if not ok_depth:
                            df = q_depth.wait_for_frame(timeout_ms // 2)
                            if df:
                                ok_depth = True
                        if ok_color and ok_depth:
                            # Optional preview
                            if show_preview and HAVE_OPENCV:
                                try:
                                    cimg = np.asanyarray(cf.get_data())
                                    dimg = np.asanyarray(df.get_data())
                                    dmap = cv2.applyColorMap(
                                        cv2.convertScaleAbs(dimg, alpha=0.03),
                                        cv2.COLORMAP_JET
                                    )
                                    max_width = 800
                                    if cimg.shape[1] > max_width:
                                        scale = max_width / cimg.shape[1]
                                        cimg = cv2.resize(cimg, None, fx=scale, fy=scale)
                                        dmap = cv2.resize(dmap, None, fx=scale, fy=scale)
                                    combined = np.hstack((cimg, dmap))
                                    cv2.imshow('Full Verification (DIRECT Color | Depth)', combined)
                                    cv2.waitKey(1)
                                except Exception:
                                    pass
                            log_with_data(logger, logging.INFO, "DIRECT mode success")
                            return FullProfileResult(cw, ch, cfps, dw, dh, dfps, True, "Success")
                    except Exception as e:
                        if first_exc is None:
                            first_exc = e
                        time.sleep(0.1)
                msg = f"No frames after {max_retries} tries: {first_exc}"
                return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, msg)
            finally:
                try:
                    if color_started:
                        color_sensor.stop()
                        time.sleep(0.15)
                except Exception:
                    pass
                try:
                    if depth_started:
                        depth_sensor.stop()
                        time.sleep(0.15)
                except Exception:
                    pass
                try:
                    if color_opened:
                        color_sensor.close()
                        time.sleep(0.1)
                except Exception:
                    pass
                try:
                    if depth_opened:
                        depth_sensor.close()
                        time.sleep(0.1)
                except Exception:
                    pass
                try:
                    log_with_data(logger, logging.DEBUG, "DIRECT stop complete")
                except Exception:
                    pass
                try:
                    del q_color
                    del q_depth
                except Exception:
                    pass
                if show_preview and HAVE_OPENCV:
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
        except Exception as e:
            if verbose:
                print("DIRECT full mode failed, falling back to pipeline:", e)
            log_with_data(logger, logging.ERROR, "DIRECT full mode failed, falling back to pipeline",
                          error=str(e), error_type=type(e).__name__)

    # Fallback: pipeline mode
    cfg = rs.config()
    cfg.enable_device(device_serial)
    cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
    pipe = rs.pipeline()

    started = False
    try:
        try:
            # Small settle before starting pipeline on macOS
            time.sleep(0.15)
            start_begin = time.time()
            pipe.start(cfg)
            started = True
            start_duration = time.time() - start_begin
            log_with_data(logger, logging.DEBUG, "Pipeline started",
                          start_duration_ms=round(start_duration * 1000, 2))
        except Exception as e:
            log_with_data(logger, logging.ERROR, "Pipeline start failed", error=str(e), error_type=type(e).__name__)
            return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, f"Start failed: {e}")

        ok = False
        first_exc = None
        for i in range(max_retries):
            try:
                frames = pipe.wait_for_frames(timeout_ms=timeout_ms)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if depth_frame and color_frame:
                    ok = True
                    log_with_data(
                        logger, logging.INFO, "First frames received (pipeline)",
                        retry_attempt=i + 1,
                        color_frame_number=color_frame.get_frame_number() if color_frame else None,
                        depth_frame_number=depth_frame.get_frame_number() if depth_frame else None,
                        color_timestamp=color_frame.get_timestamp() if color_frame else None,
                        depth_timestamp=depth_frame.get_timestamp() if depth_frame else None,
                    )

                    # Show preview if requested
                    if show_preview and HAVE_OPENCV:
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03),
                            cv2.COLORMAP_JET
                        )

                        # Resize if needed to fit on screen
                        max_width = 800
                        if color_image.shape[1] > max_width:
                            scale = max_width / color_image.shape[1]
                            color_image = cv2.resize(color_image, None, fx=scale, fy=scale)
                            depth_colormap = cv2.resize(depth_colormap, None, fx=scale, fy=scale)

                        combined = np.hstack((color_image, depth_colormap))
                        cv2.imshow('Full Verification (Color | Depth)', combined)
                        cv2.waitKey(1)

                    break
            except Exception as e:
                if first_exc is None:
                    first_exc = e
                time.sleep(0.1)

        if ok:
            log_with_data(logger, logging.INFO, "Pipeline mode success")
            return FullProfileResult(cw, ch, cfps, dw, dh, dfps, True, "Success")
        else:
            msg = f"No frames after {max_retries} tries: {first_exc}"
            log_with_data(logger, logging.WARNING, "Pipeline mode failed - no frames",
                          max_retries=max_retries,
                          first_exception=str(first_exc) if first_exc else None,
                          exception_type=type(first_exc).__name__ if first_exc else None)
            return FullProfileResult(cw, ch, cfps, dw, dh, dfps, False, msg)
    finally:
        if started:
            try:
                pipe.stop()
                time.sleep(0.25)
            except Exception:
                pass
        try:
            del pipe
            del cfg
        except Exception:
            pass
        if show_preview and HAVE_OPENCV:
            cv2.destroyAllWindows()


def verify_full_profiles_paired(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    show_preview: bool = False,
    serial: Optional[str] = None,
) -> List[FullProfileResult]:
    """
    Test paired configurations (same resolution/fps for both color and depth).
    
    This is the fastest mode and works well for most use cases.
    
    Args:
        profiles: List of (width, height, fps) tuples to test for both streams.
                 If None, uses DEFAULT_DEPTH_PROFILES (depth is more limiting).
        verbose: Whether to print progress messages.
        show_preview: Whether to show live preview window.
    
    Returns:
        List of FullProfileResult objects with test outcomes.
    """
    if profiles is None:
        profiles = DEFAULT_DEPTH_PROFILES if HAVE_VERIFIERS else []
    
    device_serial = serial or get_camera_serial()
    if not device_serial:
        if verbose:
            print("[FAIL] No RealSense device detected - skipping paired verification.")
        return []
    
    if verbose:
        print(f"\n=== Paired Mode: Testing {len(profiles)} configurations ===")
    
    results = []
    for config in profiles:
        result = verify_full_profile(
            config,
            config,
            verbose=verbose,
            show_preview=show_preview,
            serial=device_serial,
        )
        results.append(result)
        if verbose:
            print(f"  {result}")
    
    return results


def verify_full_profiles_independent(
    color_profiles: Optional[List[Tuple[int, int, int]]] = None,
    depth_profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    show_preview: bool = False,
    serial: Optional[str] = None,
) -> List[FullProfileResult]:
    """
    Test best color and best depth profiles independently, then validate together.
    
    This mode finds the best of each stream type and ensures they work together.
    
    Args:
        color_profiles: List of color (width, height, fps) tuples.
        depth_profiles: List of depth (width, height, fps) tuples.
        verbose: Whether to print progress messages.
        show_preview: Whether to show live preview window.
    
    Returns:
        List of FullProfileResult objects (typically 1-3 combinations).
    """
    if color_profiles is None:
        color_profiles = DEFAULT_COLOR_PROFILES if HAVE_VERIFIERS else []
    if depth_profiles is None:
        depth_profiles = DEFAULT_DEPTH_PROFILES if HAVE_VERIFIERS else []

    device_serial = serial or get_camera_serial()
    if not device_serial:
        if verbose:
            print("[FAIL] No RealSense device detected - skipping independent verification.")
        return []
    
    if verbose:
        print(f"\n=== Independent Mode ===")
        print(f"Finding best color profile from {len(color_profiles)} options...")
    
    # Test color profiles independently using color verifier APIs (uses DIRECT mode on macOS)
    best_color = get_best_color_profile(
        profiles=color_profiles,
        verbose=verbose,
        use_cache=False,
        validate_cached=True,
    )
    
    if not best_color:
        if verbose:
            print("  [FAIL] No working color profiles found!")
        return []
    
    if verbose:
        print(f"Finding best depth profile from {len(depth_profiles)} options...")
    
    # Test depth profiles independently using depth verifier APIs (uses DIRECT mode on macOS)
    best_depth = get_best_depth_profile(
        profiles=depth_profiles,
        verbose=verbose,
        use_cache=False,
        validate_cached=True,
    )
    
    if not best_depth:
        if verbose:
            print("  [FAIL] No working depth profiles found!")
        return []
    
    # Now test them together
    if verbose:
        print(f"\nValidating combined configuration...")
    
    result = verify_full_profile(
        best_color,
        best_depth,
        verbose=verbose,
        show_preview=show_preview,
        serial=device_serial,
    )
    return [result]


def verify_full_profiles_comprehensive(
    color_profiles: Optional[List[Tuple[int, int, int]]] = None,
    depth_profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    show_preview: bool = False,
    max_combinations: int = 100,
    serial: Optional[str] = None,
) -> List[FullProfileResult]:
    """
    Test all combinations of color × depth profiles.
    
    This is the most thorough mode but can be very slow. Use sparingly.
    
    Args:
        color_profiles: List of color (width, height, fps) tuples.
        depth_profiles: List of depth (width, height, fps) tuples.
        verbose: Whether to print progress messages.
        show_preview: Whether to show live preview window.
        max_combinations: Maximum number of combinations to test (safety limit).
    
    Returns:
        List of FullProfileResult objects with test outcomes.
    """
    if color_profiles is None:
        color_profiles = DEFAULT_COLOR_PROFILES if HAVE_VERIFIERS else []
    if depth_profiles is None:
        depth_profiles = DEFAULT_DEPTH_PROFILES if HAVE_VERIFIERS else []

    device_serial = serial or get_camera_serial()
    if not device_serial:
        if verbose:
            print("[FAIL] No RealSense device detected - skipping comprehensive verification.")
        return []
    
    total = len(color_profiles) * len(depth_profiles)
    if total > max_combinations:
        if verbose:
            print(f"Warning: {total} combinations exceeds limit of {max_combinations}")
            print(f"Testing first {max_combinations} combinations...")
    
    if verbose:
        print(f"\n=== Comprehensive Mode: Testing up to {min(total, max_combinations)} combinations ===")
    
    results = []
    count = 0
    for color_config in color_profiles:
        for depth_config in depth_profiles:
            if count >= max_combinations:
                break
            result = verify_full_profile(
                color_config,
                depth_config,
                verbose=verbose,
                show_preview=show_preview,
                serial=device_serial,
            )
            results.append(result)
            count += 1
            if verbose:
                print(f"  [{count}/{min(total, max_combinations)}] {result}")
        if count >= max_combinations:
            break
    
    return results


def get_best_full_profile(
    mode: str = "paired",
    color_profiles: Optional[List[Tuple[int, int, int]]] = None,
    depth_profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True,
    validate_cached: bool = True,
    show_preview: bool = False
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    Test full profiles and return the best available configuration.
    
    Args:
        mode: Testing mode - 'paired', 'independent', or 'comprehensive'
        color_profiles: List of color (width, height, fps) tuples.
        depth_profiles: List of depth (width, height, fps) tuples.
        verbose: Whether to print progress messages.
        use_cache: Whether to check cached profiles first.
        show_preview: Whether to show live preview window.
    
    Returns:
        ((color_w, color_h, color_fps), (depth_w, depth_h, depth_fps)) tuple,
        or None if no profiles worked.
    """
    serial: Optional[str] = None
    serial: Optional[str] = None
    # Try cached profiles first
    if use_cache:
        serial = get_camera_serial()
        if serial:
            cached = load_working_profiles(serial)
            if cached and verbose:
                print(f"Found {len(cached)} cached full profiles for camera {serial}")
                print("Validating cached best profile...")
            
            if cached:
                # Optionally validate the best cached profile; otherwise trust cache
                best_cached = cached[0]
                if not validate_cached:
                    if verbose:
                        print("Using cached full profile without validation")
                    return best_cached
                validation = verify_full_profile(
                    best_cached[0], best_cached[1],
                    verbose=verbose,
                    max_retries=5,
                    show_preview=show_preview,
                    serial=serial,
                )
                
                if validation.success:
                    if verbose:
                        print(f"[OK] Cached profile validated")
                    return best_cached
                elif verbose:
                    print("[FAIL] Cached profile failed validation, running full verification...")

    if serial is None:
        serial = get_camera_serial()

    # Run full verification based on mode
    if mode == "paired":
        profiles = depth_profiles if depth_profiles else (DEFAULT_DEPTH_PROFILES if HAVE_VERIFIERS else [])
        results = verify_full_profiles_paired(
            profiles=profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    elif mode == "independent":
        results = verify_full_profiles_independent(
            color_profiles=color_profiles,
            depth_profiles=depth_profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    elif mode == "comprehensive":
        results = verify_full_profiles_comprehensive(
            color_profiles=color_profiles,
            depth_profiles=depth_profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'paired', 'independent', or 'comprehensive'")
    
    working = [r.config_tuple for r in results if r.success]
    
    # Save working profiles
    if working:
        serial_to_save = serial or get_camera_serial()
        if serial_to_save:
            save_working_profiles(serial_to_save, working)
            if verbose:
                print(f"\n[SAVED] Saved {len(working)} working profiles for camera {serial_to_save}")
    
    return working[0] if working else None


def get_working_full_profiles(
    mode: str = "paired",
    color_profiles: Optional[List[Tuple[int, int, int]]] = None,
    depth_profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True,
    show_preview: bool = False
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """
    Test full profiles and return all working configurations.
    
    Args:
        mode: Testing mode - 'paired', 'independent', or 'comprehensive'
        color_profiles: List of color (width, height, fps) tuples.
        depth_profiles: List of depth (width, height, fps) tuples.
        verbose: Whether to print progress messages.
        use_cache: Whether to check cached profiles first.
        show_preview: Whether to show live preview window.
    
    Returns:
        List of ((color_w, color_h, color_fps), (depth_w, depth_h, depth_fps)) tuples.
    """
    serial: Optional[str] = None
    # Try cached profiles first
    if use_cache:
        serial = get_camera_serial()
        if serial:
            cached = load_working_profiles(serial)
            if cached and verbose:
                print(f"Found {len(cached)} cached full profiles for camera {serial}")
                print("Validating cached profiles...")
            
            if cached:
                # Quick validation of first few cached profiles
                validation_results = []
                for i, (color_cfg, depth_cfg) in enumerate(cached[:3]):
                    result = verify_full_profile(
                        color_cfg, depth_cfg,
                        verbose=verbose,
                        max_retries=3,
                        show_preview=show_preview,
                        serial=serial,
                    )
                    validation_results.append(result)
                
                if all(v.success for v in validation_results):
                    if verbose:
                        print(f"[OK] Cached profiles validated")
                    return cached
                elif verbose:
                    print("[FAIL] Cached profiles failed validation, running full verification...")

    if serial is None:
        serial = get_camera_serial()

    # Run full verification based on mode
    if mode == "paired":
        profiles = depth_profiles if depth_profiles else (DEFAULT_DEPTH_PROFILES if HAVE_VERIFIERS else [])
        results = verify_full_profiles_paired(
            profiles=profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    elif mode == "independent":
        results = verify_full_profiles_independent(
            color_profiles=color_profiles,
            depth_profiles=depth_profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    elif mode == "comprehensive":
        results = verify_full_profiles_comprehensive(
            color_profiles=color_profiles,
            depth_profiles=depth_profiles,
            verbose=verbose,
            show_preview=show_preview,
            serial=serial,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'paired', 'independent', or 'comprehensive'")
    
    working = [r.config_tuple for r in results if r.success]
    
    # Save working profiles
    if working:
        serial_to_save = serial or get_camera_serial()
        if serial_to_save:
            save_working_profiles(serial_to_save, working)
            if verbose:
                print(f"\n[SAVED] Saved {len(working)} working profiles for camera {serial_to_save}")
    
    return working


def main():
    """CLI entry point: runs full verification and prints results."""
    if not HAVE_REALSENSE:
        print("ERROR: pyrealsense2 is not installed")
        raise SystemExit(1)
    
    print("=" * 60)
    print("RealSense Full (Color + Depth) Profile Verification")
    print("=" * 60)
    
    # Check for cached profiles
    serial = get_camera_serial()
    if serial:
        print(f"\nCamera Serial: {serial}")
        cached = load_working_profiles(serial)
        if cached:
            print(f"\nFound {len(cached)} cached profiles.")
            # Safer default: use cache unless explicitly overridden by env or user
            import os
            env_force = os.environ.get("REALSENSE_FULL_SWEEP", "0") == "1"
            use_cache = not env_force
            if not env_force and sys.stdin and sys.stdin.isatty():
                # Prompt user whether to run a fresh sweep
                choice = input("Run full sweep now? (y/N): ").strip().lower()
                if choice == "y":
                    use_cache = False
                    print("Running fresh verification (user requested full sweep)")
            if use_cache:
                print("Using cached results (set REALSENSE_FULL_SWEEP=1 or answer 'y' to run a fresh sweep).")
                print("\n" + "=" * 60)
                print("SUMMARY (CACHED)")
                print("=" * 60)
                print(f"\n[OK] Working profiles ({len(cached)}):")
                for (cc, dc) in cached:
                    print(f"  Color:{cc[0]}x{cc[1]}@{cc[2]}fps Depth:{dc[0]}x{dc[1]}@{dc[2]}fps")
                bestc, bestd = cached[0]
                print(f"\n[BEST] Best profile:")
                print(f"  Color: {bestc[0]}x{bestc[1]}@{bestc[2]}fps")
                print(f"  Depth: {bestd[0]}x{bestd[1]}@{bestd[2]}fps")
                return
            else:
                if env_force:
                    print("Running fresh verification (REALSENSE_FULL_SWEEP=1)")
    
    # Prompt for mode
    print("\n=== Select Testing Mode ===")
    print("1. Paired - Same resolution/fps for both streams (fastest, recommended)")
    print("2. Independent - Best of each stream type (moderate)")
    print("3. Comprehensive - All combinations (slowest, most thorough)")
    print("4. Debug - Show live preview (paired mode with visualization)")
    
    mode_map = {"1": "paired", "2": "independent", "3": "comprehensive", "4": "paired"}
    show_preview_modes = {"4"}
    
    choice = input("\nSelect mode (1-4, default=1): ").strip() or "1"
    if choice not in mode_map:
        print("Invalid choice, using paired mode")
        choice = "1"
    
    mode = mode_map[choice]
    show_preview = choice in show_preview_modes
    
    print(f"\nRunning {mode} mode verification...")
    
    try:
        if mode == "paired":
            results = verify_full_profiles_paired(
                verbose=True,
                show_preview=show_preview,
                serial=serial,
            )
        elif mode == "independent":
            results = verify_full_profiles_independent(
                verbose=True,
                show_preview=show_preview,
                serial=serial,
            )
        elif mode == "comprehensive":
            results = verify_full_profiles_comprehensive(
                verbose=True,
                show_preview=show_preview,
                serial=serial,
            )
    except Exception as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if working:
        print(f"\n[OK] Working profiles ({len(working)}):")
        for result in working:
            print(f"  Color:{result.color_width}x{result.color_height}@{result.color_fps}fps "
                  f"Depth:{result.depth_width}x{result.depth_height}@{result.depth_fps}fps")
        
        best = working[0]
        print(f"\n[BEST] Best profile:")
        print(f"  Color: {best.color_width}x{best.color_height}@{best.color_fps}fps")
        print(f"  Depth: {best.depth_width}x{best.depth_height}@{best.depth_fps}fps")
        
        # Save profiles
        if serial:
            save_working_profiles(serial, [r.config_tuple for r in working])
            print(f"\n[SAVED] Saved working profiles for camera {serial}")
    else:
        print("\n[FAIL] No working profiles found!")
    
    if failed:
        print(f"\n[FAIL] Failed profiles ({len(failed)}):")
        for result in failed[:5]:  # Show first 5 failures
            print(f"  {result}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    # Exit non-zero if none worked
    if not working:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
