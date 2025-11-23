'''
RealSense D400 Series Depth Profile Verification Tool
Tests multiple depth stream configurations to identify which produce valid frames.

Provides both CLI tool and API for programmatic use.

@aldrick-t, 2025
'''

from __future__ import annotations
import os
import sys
import time
import logging
import json
import platform
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

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
CONFIG_NAMESPACE = "realsense_verify_depth"

# Setup logging
def setup_logger(log_file: Path) -> logging.Logger:
    """Setup structured JSON logger for depth verification."""
    logger = logging.getLogger("realsense_depth_verify")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler - JSON formatted, DEBUG level
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
    """Log message with additional structured data."""
    extra = {'extra_data': kwargs} if kwargs else {}
    logger.log(level, message, extra=extra)

def get_system_info() -> Dict[str, Any]:
    """Gather system information for diagnostics."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    
    if HAVE_PSUTIL:
        try:
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
            info["memory_percent"] = mem.percent
        except Exception as e:
            info["psutil_error"] = str(e)
    
    return info

def get_camera_info(device) -> Dict[str, Any]:
    """Extract detailed camera information."""
    info = {}
    try:
        info["name"] = device.get_info(rs.camera_info.name)
        info["serial_number"] = device.get_info(rs.camera_info.serial_number)
        info["firmware_version"] = device.get_info(rs.camera_info.firmware_version)
        info["physical_port"] = device.get_info(rs.camera_info.physical_port)
        info["product_id"] = device.get_info(rs.camera_info.product_id)
        info["usb_type_descriptor"] = device.get_info(rs.camera_info.usb_type_descriptor)
        
        # Try to get additional info
        try:
            info["recommended_firmware"] = device.get_info(rs.camera_info.recommended_firmware_version)
        except:
            pass
        
    except Exception as e:
        info["error"] = str(e)
    
    return info

def get_sensor_info(sensor) -> Dict[str, Any]:
    """Extract sensor capabilities and options."""
    info = {"name": "unknown", "supported_options": {}}
    
    try:
        info["name"] = sensor.get_info(rs.camera_info.name)
    except:
        pass
    
    # Get supported options
    try:
        for opt in [rs.option.emitter_enabled, rs.option.laser_power,
                    rs.option.exposure, rs.option.gain, rs.option.enable_auto_exposure,
                    rs.option.visual_preset, rs.option.depth_units]:
            try:
                if sensor.supports(opt):
                    opt_range = sensor.get_option_range(opt)
                    info["supported_options"][opt.name] = {
                        "min": opt_range.min,
                        "max": opt_range.max,
                        "step": opt_range.step,
                        "default": opt_range.default,
                        "current": sensor.get_option(opt)
                    }
            except:
                pass
    except:
        pass
    
    return info



# Extended list of (width, height, fps) to try. Ordered high→low bandwidth.
# Includes standard resolutions, square formats, and various framerates.
DEFAULT_DEPTH_PROFILES = [
    # High resolution - 16:9 and square
    (1280, 720, 30),
    (1280, 720, 15),
    (1280, 720, 6),
    (720, 720, 30),   # Square
    (720, 720, 15),   # Square
    
    # Medium-high resolution
    (848, 480, 90),
    (848, 480, 60),
    (848, 480, 30),
    (848, 480, 15),
    (848, 480, 6),
    (480, 480, 60),   # Square
    (480, 480, 30),   # Square
    (480, 480, 15),   # Square
    
    # Standard VGA resolution
    (640, 480, 90),
    (640, 480, 60),
    (640, 480, 30),
    (640, 480, 15),
    (640, 480, 6),
    
    # Low resolution
    (424, 240, 90),
    (424, 240, 60),
    (424, 240, 30),
    (424, 240, 15),
    (424, 240, 6),
    (240, 240, 60),   # Square
    (240, 240, 30),   # Square
    
    # Very low resolution (fallback)
    (320, 240, 30),
    (320, 240, 15),
    (320, 240, 6),
]


@dataclass
class DepthProfileResult:
    """Result of testing a depth profile configuration."""
    width: int
    height: int
    fps: int
    success: bool
    message: str
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        return (self.width, self.height)
    
    @property
    def config_tuple(self) -> Tuple[int, int, int]:
        """Get full configuration as (width, height, fps) tuple."""
        return (self.width, self.height, self.fps)
    
    def __repr__(self) -> str:
        status = "[OK]" if self.success else "[FAIL]"
        return f"{status} {self.width}x{self.height}@{self.fps}fps: {self.message}"


def get_camera_serial() -> Optional[str]:
    """Get the serial number of the connected RealSense camera."""
    if not HAVE_REALSENSE:
        return None
    
    try:
        ctx = rs.context()
        devices = ctx.devices
        if len(devices) > 0:
            return devices[0].get_info(rs.camera_info.serial_number)
    except Exception:
        pass
    return None


def save_working_profiles(serial: str, profiles: List[Tuple[int, int, int]]):
    """Save working depth profiles for a specific camera serial."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    if "camera_profiles" not in ns:
        ns["camera_profiles"] = {}
    
    ns["camera_profiles"][serial] = {
        "profiles": [list(p) for p in profiles],
        "last_verified": time.time(),
    }
    
    save_config(cfg)


def load_working_profiles(serial: str) -> Optional[List[Tuple[int, int, int]]]:
    """Load saved working depth profiles for a specific camera serial."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    camera_profiles = ns.get("camera_profiles", {})
    if serial not in camera_profiles:
        return None
    
    profiles_data = camera_profiles[serial].get("profiles", [])
    return [tuple(p) for p in profiles_data]


def verify_depth_profiles(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    timeout_ms: int = 1500,
    max_retries: int = 10
) -> List[DepthProfileResult]:
    """
    Test multiple depth stream configurations and return results.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_DEPTH_PROFILES.
        verbose: Whether to print progress messages.
        timeout_ms: Timeout in milliseconds for wait_for_frames.
        max_retries: Maximum number of frame read attempts per profile.
    
    Returns:
        List of DepthProfileResult objects with test outcomes.
    
    Raises:
        RuntimeError: If pyrealsense2 is not available or no device found.
    """
    if not HAVE_REALSENSE:
        raise RuntimeError("pyrealsense2 is required but not installed")
    
    if profiles is None:
        profiles = DEFAULT_DEPTH_PROFILES

    # Setup logging
    log_path = REPO_ROOT / "pickafresa_vision" / "logs" / "realsense_depth_verify.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    logger = setup_logger(log_path)

    # System info
    system_info = get_system_info()
    log_with_data(logger, logging.INFO, "Starting depth profile verification",
                  system_info=system_info, profiles_count=len(profiles))

    # Initialize context and verify device
    ctx = rs.context()
    devices = ctx.devices

    device_names = [d.get_info(rs.camera_info.name) for d in devices] if len(devices) > 0 else []
    log_with_data(logger, logging.INFO, "RealSense context initialized",
                  devices_found=len(devices), device_names=device_names)
    if verbose:
        print("Devices:", device_names)

    if len(devices) == 0:
        log_with_data(logger, logging.ERROR, "No RealSense device found")
        raise RuntimeError("No RealSense device found")

    dev = devices[0]

    # Resolve device serial once and bind all configs to this device
    device_serial = None
    try:
        device_serial = dev.get_info(rs.camera_info.serial_number)
        log_with_data(logger, logging.DEBUG, "Device serial resolved", serial=device_serial)
    except Exception as e:
        device_serial = None
        log_with_data(logger, logging.WARNING, "Failed to get device serial", error=str(e))

    # Camera info
    camera_info = get_camera_info(dev)
    log_with_data(logger, logging.INFO, "Camera detected", camera_info=camera_info)

    # Try to enable the emitter when available (helps depth on some units)
    try:
        stereo = next(s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name))
        if stereo.supports(rs.option.emitter_enabled):
            stereo.set_option(rs.option.emitter_enabled, 1)
            log_with_data(logger, logging.INFO, "Emitter enabled")
            if verbose:
                print("Emitter enabled")
    except Exception as e:
        log_with_data(logger, logging.WARNING, "Emitter setup skipped", error=str(e), error_type=type(e).__name__)
        if verbose:
            print("Emitter setup skipped:", e)

    results = []

    # On macOS, avoid pipeline/device_hub when possible. Use direct depth sensor open/start.
    use_pipeline = os.environ.get("REALSENSE_DEPTH_USE_PIPELINE", "0") == "1"
    log_with_data(logger, logging.INFO, "Mode selection",
                  platform=sys.platform, use_pipeline=use_pipeline,
                  reason="env_override" if use_pipeline else "direct_mode_preferred")
    if sys.platform == "darwin" and not use_pipeline:
        if verbose:
            print("Using DIRECT depth sensor mode (macOS) to avoid pipeline power toggles")
        log_with_data(logger, logging.INFO, "Entering DIRECT depth sensor mode")
        try:
            ctx2 = rs.context()
            dev2 = None
            for d in ctx2.devices:
                if (not device_serial) or d.get_info(rs.camera_info.serial_number) == device_serial:
                    dev2 = d
                    break
            if dev2 is None:
                log_with_data(logger, logging.ERROR, "No device matched for direct mode", target_serial=device_serial)
                raise RuntimeError("No RealSense device matched for direct mode")

            # Find a sensor that exposes depth stream profiles
            depth_sensor = None
            for s in dev2.query_sensors():
                try:
                    s_profiles = s.get_stream_profiles()
                except Exception:
                    continue
                if any(p.stream_type() == rs.stream.depth for p in s_profiles):
                    depth_sensor = s
                    break
            if depth_sensor is None:
                log_with_data(logger, logging.ERROR, "No depth sensor found on device")
                raise RuntimeError("No depth sensor found on device")

            for (w, h, fps) in profiles:
                if verbose:
                    print(f"\n=== Testing depth (DIRECT) {w}x{h}@{fps} ===")
                # Select a matching stream profile (z16)
                match = None
                available_profiles = []
                try:
                    for p in depth_sensor.get_stream_profiles():
                        try:
                            if p.stream_type() != rs.stream.depth:
                                continue
                            vsp = p.as_video_stream_profile()
                            if not vsp:
                                continue
                            prof_desc = f"{vsp.width()}x{vsp.height()}@{p.fps()}fps({p.format()})"
                            available_profiles.append(prof_desc)
                            if vsp.width() == w and vsp.height() == h and p.fps() == fps and p.format() == rs.format.z16:
                                match = p
                                break
                        except Exception:
                            continue
                except Exception as e:
                    log_with_data(logger, logging.WARNING, "Could not enumerate depth profiles",
                                  error=str(e), error_type=type(e).__name__)
                    if verbose:
                        print("Could not enumerate depth profiles:", e)

                if match is None:
                    msg = "No matching depth profile on sensor"
                    results.append(DepthProfileResult(w, h, fps, False, msg))
                    log_with_data(logger, logging.WARNING, msg, requested_profile={"width": w, "height": h, "fps": fps},
                                  available_profiles=available_profiles[:10])
                    if verbose:
                        print("[FAIL]", msg)
                    continue

                q = rs.frame_queue(1)
                opened = False
                started = False
                open_time = None
                start_time_sensor = None
                test_begin = time.time()
                try:
                    log_with_data(logger, logging.DEBUG, "Opening depth sensor", width=w, height=h, fps=fps)
                    time.sleep(0.1)
                    open_start = time.time()
                    depth_sensor.open(match)
                    opened = True
                    open_time = time.time() - open_start
                    log_with_data(logger, logging.DEBUG, "Depth sensor opened", open_duration_ms=round(open_time * 1000, 2))
                    time.sleep(0.05)
                    start_begin = time.time()
                    depth_sensor.start(q)
                    started = True
                    start_time_sensor = time.time() - start_begin
                    log_with_data(logger, logging.DEBUG, "Depth sensor started", start_duration_ms=round(start_time_sensor * 1000, 2))

                    ok = False
                    first_exc = None
                    retry_times = []
                    for _ in range(max_retries):
                        try:
                            frame = q.wait_for_frame(timeout_ms)
                            if frame:
                                ok = True
                                total_time = time.time() - test_begin
                                log_with_data(logger, logging.INFO, "First depth frame received",
                                              total_latency_ms=round(total_time * 1000, 2),
                                              frame_number=frame.get_frame_number(), timestamp=frame.get_timestamp())
                                break
                        except Exception as e:
                            if first_exc is None:
                                first_exc = e
                            retry_times.append(round((time.time() - test_begin) * 1000, 2))
                            time.sleep(0.1)

                    if ok:
                        results.append(DepthProfileResult(w, h, fps, True, "ok"))
                        log_with_data(logger, logging.INFO, "Depth profile test successful", width=w, height=h, fps=fps,
                                      open_duration_ms=round(open_time * 1000, 2) if open_time else None,
                                      start_duration_ms=round(start_time_sensor * 1000, 2) if start_time_sensor else None)
                    else:
                        msg = f"no frame within timeout; first_exc={first_exc}"
                        if verbose:
                            print("[FAIL]", msg)
                        log_with_data(logger, logging.WARNING, "Depth profile test failed",
                                      width=w, height=h, fps=fps, max_retries=max_retries,
                                      first_exception=str(first_exc) if first_exc else None,
                                      exception_type=type(first_exc).__name__ if first_exc else None,
                                      retry_durations_ms=retry_times)
                        results.append(DepthProfileResult(w, h, fps, False, msg))
                finally:
                    try:
                        if started:
                            depth_sensor.stop()
                            log_with_data(logger, logging.DEBUG, "Depth sensor stopped")
                            time.sleep(0.15)
                    except Exception:
                        pass
                    try:
                        if opened:
                            depth_sensor.close()
                            log_with_data(logger, logging.DEBUG, "Depth sensor closed")
                            time.sleep(0.1)
                    except Exception:
                        pass
                    try:
                        del q
                    except Exception:
                        pass

            log_with_data(logger, logging.INFO, "DIRECT mode testing complete", profiles_tested=len(profiles),
                          successful=sum(1 for r in results if r.success))
            return results
        except Exception as e:
            log_with_data(logger, logging.ERROR, "DIRECT depth mode failed, falling back to pipeline",
                          error=str(e), error_type=type(e).__name__)
            if verbose:
                print("DIRECT depth mode failed, falling back to pipeline:", e)
    
    # Reuse a single pipeline instance to avoid repeated device hub allocations
    log_with_data(logger, logging.INFO, "Using pipeline mode for depth testing")
    shared_pipe = rs.pipeline()

    for (w, h, fps) in profiles:
        test_begin = time.time()
        cfg = rs.config()
        if device_serial:
            cfg.enable_device(device_serial)
            log_with_data(logger, logging.DEBUG, "Pipeline configured with device serial", serial=device_serial)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        
        if verbose:
            print(f"\n=== Testing depth {w}x{h}@{fps} ===")
        
        started = False
        start_duration = None
        try:
            try:
                # Small settle before starting
                time.sleep(0.15)
                start_begin = time.time()
                shared_pipe.start(cfg)
                start_duration = time.time() - start_begin
                started = True
                log_with_data(logger, logging.DEBUG, "Pipeline started", start_duration_ms=round(start_duration * 1000, 2))
            except Exception as e:
                if verbose:
                    print("Start failed:", e)
                log_with_data(logger, logging.ERROR, "Pipeline start failed", error=str(e), error_type=type(e).__name__)
                results.append(DepthProfileResult(w, h, fps, False, f"start: {e}"))
                continue
            
            ok = False
            first_exc = None
            retry_times = []
            for i in range(max_retries):
                try:
                    frames = shared_pipe.wait_for_frames(timeout_ms)
                    d = frames.get_depth_frame() if frames else None
                    if d:
                        if verbose:
                            print("[OK] Depth frame received")
                        total_latency = time.time() - test_begin
                        log_with_data(logger, logging.INFO, "First depth frame (pipeline)",
                                      retry_attempt=i + 1,
                                      total_latency_ms=round(total_latency * 1000, 2),
                                      frame_number=d.get_frame_number(), timestamp=d.get_timestamp())
                        ok = True
                        break
                except Exception as e:
                    if first_exc is None:
                        first_exc = e
                    retry_times.append(round((time.time() - test_begin) * 1000, 2))
                    time.sleep(0.2)
            
            if ok:
                results.append(DepthProfileResult(w, h, fps, True, "ok"))
                log_with_data(logger, logging.INFO, "Depth profile test successful (pipeline)", width=w, height=h, fps=fps,
                              start_duration_ms=round(start_duration * 1000, 2) if start_duration else None)
            else:
                msg = f"no frame within timeout; first_exc={first_exc}"
                if verbose:
                    print("FAIL:", msg)
                log_with_data(logger, logging.WARNING, "Depth test failed - no frames (pipeline)", width=w, height=h, fps=fps,
                              max_retries=max_retries, first_exception=str(first_exc) if first_exc else None,
                              exception_type=type(first_exc).__name__ if first_exc else None,
                              retry_durations_ms=retry_times)
                results.append(DepthProfileResult(w, h, fps, False, msg))
        finally:
            if started:
                try:
                    shared_pipe.stop()
                    log_with_data(logger, logging.DEBUG, "Pipeline stopped")
                    time.sleep(0.25)
                except Exception:
                    pass
            # Explicitly drop config to encourage immediate teardown
            try:
                del cfg
            except Exception:
                pass
    
    log_with_data(logger, logging.INFO, "Pipeline mode testing complete",
                  profiles_tested=len(profiles), successful=sum(1 for r in results if r.success))
    return results


def get_best_depth_profile(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True,
    validate_cached: bool = True,
) -> Optional[Tuple[int, int, int]]:
    """
    Test depth profiles and return the best available configuration.
    
    The "best" profile is the first successful one in the provided list,
    which should be ordered from highest to lowest quality/bandwidth.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_DEPTH_PROFILES.
        verbose: Whether to print progress messages.
        use_cache: Whether to check cached profiles first.
    
    Returns:
        (width, height, fps) tuple of the best working profile,
        or None if no profiles worked.
    """
    # Try cached profiles first
    if use_cache:
        serial = get_camera_serial()
        if serial:
            cached = load_working_profiles(serial)
            if cached and verbose:
                print(f"Found {len(cached)} cached profiles for camera {serial}")
                print("Validating cached best profile...")
            
            if cached:
                # Optionally validate the best cached profile; otherwise trust cache
                best_cached = cached[0]
                if not validate_cached:
                    if verbose:
                        print(f"Using cached profile without validation: {best_cached}")
                    return best_cached
                validation = verify_depth_profiles(
                    profiles=[best_cached],
                    verbose=verbose,
                    max_retries=5
                )
                
                if validation and validation[0].success:
                    if verbose:
                        print(f"[OK] Cached profile validated: {best_cached[0]}x{best_cached[1]}@{best_cached[2]}fps")
                    return best_cached
                elif verbose:
                    print("[FAIL] Cached profile failed validation, running full verification...")
    
    # Run full verification
    results = verify_depth_profiles(profiles=profiles, verbose=verbose)
    
    working = [r.config_tuple for r in results if r.success]
    
    # Save working profiles
    if working:
        serial = get_camera_serial()
        if serial:
            save_working_profiles(serial, working)
            if verbose:
                print(f"\nSaved {len(working)} working profiles for camera {serial}")
    
    return working[0] if working else None


def get_working_depth_profiles(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True
) -> List[Tuple[int, int, int]]:
    """
    Test depth profiles and return all working configurations.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_DEPTH_PROFILES.
        verbose: Whether to print progress messages.
        use_cache: Whether to check cached profiles first.
    
    Returns:
        List of (width, height, fps) tuples for all successful profiles.
    """
    # Try cached profiles first
    if use_cache:
        serial = get_camera_serial()
        if serial:
            cached = load_working_profiles(serial)
            if cached and verbose:
                print(f"Found {len(cached)} cached profiles for camera {serial}")
                print("Validating cached profiles...")
            
            if cached:
                # Quick validation of first few cached profiles
                validation = verify_depth_profiles(
                    profiles=cached[:3],
                    verbose=verbose,
                    max_retries=3
                )
                
                if all(v.success for v in validation):
                    if verbose:
                        print(f"[OK] Cached profiles validated")
                    return cached
                elif verbose:
                    print("[FAIL] Cached profiles failed validation, running full verification...")
    
    # Run full verification
    results = verify_depth_profiles(profiles=profiles, verbose=verbose)
    working = [r.config_tuple for r in results if r.success]
    
    # Save working profiles
    if working:
        serial = get_camera_serial()
        if serial:
            save_working_profiles(serial, working)
            if verbose:
                print(f"\nSaved {len(working)} working profiles for camera {serial}")
    
    return working


def main():
    """CLI entry point: runs full verification sweep and prints results."""
    if not HAVE_REALSENSE:
        print("ERROR: pyrealsense2 is not installed")
        raise SystemExit(1)
    
    print("=" * 60)
    print("RealSense Depth Profile Verification")
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
                for (w, h, fps) in cached:
                    print(f"  {w}x{h}@{fps}fps")
                best = cached[0]
                print(f"\n[BEST] Best profile: {best[0]}x{best[1]}@{best[2]}fps")
                return
            else:
                if env_force:
                    print("Running fresh verification (REALSENSE_FULL_SWEEP=1)")
    
    try:
        results = verify_depth_profiles(verbose=True)
    except Exception as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = []
    failed = []
    
    for result in results:
        if result.success:
            working.append(result)
        else:
            failed.append(result)
    
    if working:
        print(f"\n[OK] Working profiles ({len(working)}):")
        for result in working:
            print(f"  {result.width}x{result.height}@{result.fps}fps")
        
        best = working[0]
        print(f"\n[BEST] Best profile: {best.width}x{best.height}@{best.fps}fps")
        
        # Save profiles
        if serial:
            save_working_profiles(serial, [r.config_tuple for r in working])
            print(f"\n[SAVED] Saved working profiles for camera {serial}")
    else:
        print("\n[FAIL] No working profiles found!")
    
    if failed:
        print(f"\n[FAIL] Failed profiles ({len(failed)}):")
        for result in failed[:5]:  # Show first 5 failures
            print(f"  {result.width}x{result.height}@{result.fps}fps - {result.message}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    # Exit non-zero if none worked
    if not working:
        raise SystemExit(1)


if __name__ == "__main__":
    main()