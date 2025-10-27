'''
RealSense D400 Series Color Stream Verification Tool
Tests multiple color stream configurations to identify which produce valid frames.

Provides both CLI tool and API for programmatic use.

Team YEA, 2025
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
CONFIG_NAMESPACE = "realsense_verify_color"

# Setup logging
def setup_logger(log_file: Path) -> logging.Logger:
    """Setup structured JSON logger for color verification."""
    logger = logging.getLogger("realsense_color_verify")
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
        for opt in [rs.option.exposure, rs.option.gain, rs.option.enable_auto_exposure,
                    rs.option.brightness, rs.option.contrast, rs.option.gamma,
                    rs.option.hue, rs.option.saturation, rs.option.sharpness,
                    rs.option.white_balance, rs.option.enable_auto_white_balance,
                    rs.option.backlight_compensation, rs.option.power_line_frequency]:
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
# More aggressive than depth profiles since color streams have higher bandwidth tolerance.
DEFAULT_COLOR_PROFILES = [
    # Ultra high resolution - 16:9
    (1920, 1080, 30),
    (1920, 1080, 15),
    (1920, 1080, 6),
    
    # High resolution - 16:9 and square
    (1280, 720, 60),
    (1280, 720, 30),
    (1280, 720, 15),
    (1280, 720, 6),
    (720, 720, 60),   # Square
    (720, 720, 30),   # Square
    (720, 720, 15),   # Square
    
    # Medium-high resolution
    (848, 480, 90),
    (848, 480, 60),
    (848, 480, 30),
    (848, 480, 15),
    (848, 480, 6),
    (480, 480, 90),   # Square
    (480, 480, 60),   # Square
    (480, 480, 30),   # Square
    (480, 480, 15),   # Square
    
    # Standard VGA resolution
    (640, 480, 90),
    (640, 480, 60),
    (640, 480, 30),
    (640, 480, 15),
    (640, 480, 6),
    
    # 640p Square
    (640, 640, 90),
    (640, 640, 60),
    (640, 640, 30),
    (640, 640, 15),
    (640, 640, 6),
    
    # Low resolution
    (424, 240, 90),
    (424, 240, 60),
    (424, 240, 30),
    (424, 240, 15),
    (424, 240, 6),
    (240, 240, 90),   # Square
    (240, 240, 60),   # Square
    (240, 240, 30),   # Square
    
    # Very low resolution (fallback)
    (320, 240, 60),
    (320, 240, 30),
    (320, 240, 15),
    (320, 240, 6),
]


@dataclass
class ColorProfileResult:
    """Result of testing a color profile configuration."""
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
    """Save working color profiles for a specific camera serial."""
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
    """Load saved working color profiles for a specific camera serial."""
    cfg = load_config()
    ns = get_namespace(cfg, CONFIG_NAMESPACE)
    
    camera_profiles = ns.get("camera_profiles", {})
    if serial not in camera_profiles:
        return None
    
    profiles_data = camera_profiles[serial].get("profiles", [])
    return [tuple(p) for p in profiles_data]


def verify_color_profiles(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True,
    timeout_ms: int = 1500,
    max_retries: int = 10
) -> List[ColorProfileResult]:
    """
    Test multiple color stream configurations and return results.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_COLOR_PROFILES.
        verbose: Whether to print progress messages.
        timeout_ms: Timeout in milliseconds for wait_for_frames.
        max_retries: Maximum number of frame read attempts per profile.
    
    Returns:
        List of ColorProfileResult objects with test outcomes.
    
    Raises:
        RuntimeError: If pyrealsense2 is not available or no device found.
    """
    if not HAVE_REALSENSE:
        raise RuntimeError("pyrealsense2 is required but not installed")
    
    if profiles is None:
        profiles = DEFAULT_COLOR_PROFILES
    
    # Setup logging
    log_path = REPO_ROOT / "pickafresa_vision" / "logs" / "realsense_color_verify.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    logger = setup_logger(log_path)
    
    # Log system information
    system_info = get_system_info()
    log_with_data(logger, logging.INFO, "Starting color profile verification", 
                  system_info=system_info, profiles_count=len(profiles))
    
    # Initialize context and verify device
    ctx = rs.context()
    devices = ctx.devices
    
    log_with_data(logger, logging.INFO, "RealSense context initialized",
                  devices_found=len(devices),
                  device_names=[d.get_info(rs.camera_info.name) for d in devices] if len(devices) > 0 else [])
    
    if verbose:
        print("Devices:", [d.get_info(rs.camera_info.name) for d in devices])
    
    if len(devices) == 0:
        log_with_data(logger, logging.ERROR, "No RealSense device found")
        raise RuntimeError("No RealSense device found")
    
    # Get detailed camera information
    camera_info = get_camera_info(devices[0])
    log_with_data(logger, logging.INFO, "Camera detected", camera_info=camera_info)
    
    results = []
    
    # Resolve device serial once and bind all configs to this device to avoid
    # librealsense probing unintended sensors and to minimize re-enumeration.
    device_serial = None
    try:
        device_serial = devices[0].get_info(rs.camera_info.serial_number) if len(devices) > 0 else None
        log_with_data(logger, logging.DEBUG, "Device serial resolved", serial=device_serial)
    except Exception as e:
        device_serial = None
        log_with_data(logger, logging.WARNING, "Failed to get device serial", error=str(e))

    # On macOS, avoid pipeline/device_hub when possible. Use direct RGB sensor open/start.
    use_pipeline = os.environ.get("REALSENSE_COLOR_USE_PIPELINE", "0") == "1"
    log_with_data(logger, logging.INFO, "Mode selection", 
                  platform=sys.platform, 
                  use_pipeline=use_pipeline,
                  reason="env_override" if use_pipeline else "direct_mode_preferred")
    
    if sys.platform == "darwin" and not use_pipeline:
        if verbose:
            print("Using DIRECT sensor mode (macOS) to avoid pipeline power toggles")
        try:
            ctx2 = rs.context()
            dev2 = None
            for d in ctx2.devices:
                if (not device_serial) or d.get_info(rs.camera_info.serial_number) == device_serial:
                    dev2 = d
                    break
            if dev2 is None:
                raise RuntimeError("No RealSense device matched for direct mode")
            # Find the color/RGB sensor
            color_sensor = None
            for s in dev2.query_sensors():
                try:
                    name = s.get_info(rs.camera_info.name)
                except Exception:
                    name = ""
                # Prefer sensors that explicitly support color stream
                if any(p.stream_type() == rs.stream.color for p in s.get_stream_profiles()):
                    color_sensor = s
                    break
                if "RGB" in name or "Color" in name:
                    color_sensor = s
                    break
            if color_sensor is None:
                raise RuntimeError("No color sensor found on device")

            for (w, h, fps) in profiles:
                if verbose:
                    print(f"\n=== Testing color (DIRECT) {w}x{h}@{fps} ===")
                # Select a matching stream profile
                match = None
                try:
                    for p in color_sensor.get_stream_profiles():
                        try:
                            if p.stream_type() != rs.stream.color:
                                continue
                            vsp = p.as_video_stream_profile()
                            if not vsp:
                                continue
                            if vsp.width() == w and vsp.height() == h and p.fps() == fps and \
                               p.format() in (rs.format.bgr8, rs.format.rgb8, rs.format.yuyv):
                                match = p
                                break
                        except Exception:
                            continue
                except Exception as e:
                    if verbose:
                        print("Could not enumerate stream profiles:", e)

                if match is None:
                    msg = "No matching stream profile on sensor"
                    results.append(ColorProfileResult(w, h, fps, False, msg))
                    if verbose:
                        print("[FAIL]", msg)
                    continue

                # Start sensor with a frame_queue
                q = rs.frame_queue(1)
                opened = False
                started = False
                try:
                    try:
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(f"DIRECT OPEN/START {w}x{h}@{fps} at {time.time():.3f}\n")
                    except Exception:
                        pass

                    time.sleep(0.1)
                    color_sensor.open(match)
                    opened = True
                    time.sleep(0.05)
                    color_sensor.start(q)
                    started = True

                    ok = False
                    first_exc = None
                    for _ in range(max_retries):
                        try:
                            frame = q.wait_for_frame(timeout_ms)
                            if frame:
                                ok = True
                                break
                        except Exception as e:
                            if first_exc is None:
                                first_exc = e
                            time.sleep(0.1)
                    if ok:
                        results.append(ColorProfileResult(w, h, fps, True, "Success"))
                        if verbose:
                            print("[OK] Color frames received")
                    else:
                        msg = f"No frames after {max_retries} tries: {first_exc}"
                        results.append(ColorProfileResult(w, h, fps, False, msg))
                        if verbose:
                            print("[FAIL]", msg)
                finally:
                    try:
                        if started:
                            color_sensor.stop()
                            time.sleep(0.15)
                    except Exception:
                        pass
                    try:
                        if opened:
                            color_sensor.close()
                            time.sleep(0.1)
                    except Exception:
                        pass
                    try:
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(f"DIRECT STOP {w}x{h}@{fps} at {time.time():.3f}\n")
                    except Exception:
                        pass
                    try:
                        del q
                    except Exception:
                        pass

            return results
        except Exception as e:
            if verbose:
                print("DIRECT mode failed, falling back to pipeline:", e)

    # Reuse a single pipeline instance to avoid repeated device hub allocations
    log_with_data(logger, logging.INFO, "Using pipeline mode for stream testing")
    shared_pipe = rs.pipeline()

    for (w, h, fps) in profiles:
        start_time = time.time()
        profile_config = {"width": w, "height": h, "fps": fps}
        
        log_with_data(logger, logging.INFO, "Testing profile (pipeline mode)", **profile_config)
        
        cfg = rs.config()
        if device_serial:
            cfg.enable_device(device_serial)
            log_with_data(logger, logging.DEBUG, "Pipeline configured with device serial", serial=device_serial)
        
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        
        if verbose:
            print(f"\n=== Testing color {w}x{h}@{fps} ===")
        
        started = False
        pipeline_start_time = None
        first_frame_time = None
        
        try:
            try:
                # Give the USB stack a brief moment to settle before (re)starting
                log_with_data(logger, logging.DEBUG, "Settling before pipeline start", delay_ms=150)
                time.sleep(0.15)
                
                start_begin = time.time()
                shared_pipe.start(cfg)
                pipeline_start_time = time.time() - start_begin
                started = True
                
                log_with_data(logger, logging.DEBUG, "Pipeline started successfully", 
                              start_duration_ms=round(pipeline_start_time * 1000, 2))
            except Exception as e:
                log_with_data(logger, logging.ERROR, "Pipeline start failed", 
                              **profile_config,
                              error=str(e),
                              error_type=type(e).__name__)
                results.append(ColorProfileResult(w, h, fps, False, f"Start failed: {e}"))
                if verbose:
                    print(f"[FAIL] Could not start: {e}")
                continue
            
            ok = False
            first_exc = None
            retry_times = []
            
            for i in range(max_retries):
                retry_start = time.time()
                try:
                    frames = shared_pipe.wait_for_frames(timeout_ms=timeout_ms)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        ok = True
                        first_frame_time = time.time()
                        frame_latency = first_frame_time - start_time
                        
                        log_with_data(logger, logging.INFO, "First frame received (pipeline)", 
                                      retry_attempt=i + 1,
                                      total_latency_ms=round(frame_latency * 1000, 2),
                                      frame_number=color_frame.get_frame_number(),
                                      timestamp=color_frame.get_timestamp())
                        break
                except Exception as e:
                    retry_duration = time.time() - retry_start
                    retry_times.append(round(retry_duration * 1000, 2))
                    
                    if first_exc is None:
                        first_exc = e
                    
                    log_with_data(logger, logging.DEBUG, "Frame wait failed (pipeline)", 
                                  retry_attempt=i + 1,
                                  error=str(e),
                                  error_type=type(e).__name__,
                                  retry_duration_ms=retry_times[-1])
                    time.sleep(0.1)
            
            if ok:
                total_time = time.time() - start_time
                results.append(ColorProfileResult(w, h, fps, True, "Success"))
                
                log_with_data(logger, logging.INFO, "Profile test successful (pipeline)", 
                              **profile_config,
                              total_test_duration_ms=round(total_time * 1000, 2),
                              pipeline_start_duration_ms=round(pipeline_start_time * 1000, 2) if pipeline_start_time else None)
                
                if verbose:
                    print(f"[OK] Color frames received")
            else:
                total_time = time.time() - start_time
                msg = f"No frames after {max_retries} tries: {first_exc}"
                results.append(ColorProfileResult(w, h, fps, False, msg))
                
                log_with_data(logger, logging.WARNING, "Profile test failed - no frames (pipeline)", 
                              **profile_config,
                              max_retries=max_retries,
                              first_exception=str(first_exc),
                              exception_type=type(first_exc).__name__ if first_exc else None,
                              retry_durations_ms=retry_times,
                              total_test_duration_ms=round(total_time * 1000, 2))
                
                if verbose:
                    print(f"[FAIL] {msg}")
        finally:
            if started:
                try:
                    log_with_data(logger, logging.DEBUG, "Stopping pipeline")
                    stop_start = time.time()
                    shared_pipe.stop()
                    stop_time = time.time() - stop_start
                    log_with_data(logger, logging.DEBUG, "Pipeline stopped", 
                                  stop_duration_ms=round(stop_time * 1000, 2))
                    # Allow hardware to settle between rapid start/stop cycles on macOS
                    time.sleep(0.25)
                except Exception as e:
                    log_with_data(logger, logging.WARNING, "Error stopping pipeline", 
                                  error=str(e), error_type=type(e).__name__)
            
            # Explicitly drop config to encourage immediate teardown of stream request
            try:
                del cfg
            except Exception:
                pass
    
    log_with_data(logger, logging.INFO, "Pipeline mode testing complete", 
                  profiles_tested=len(profiles),
                  successful=sum(1 for r in results if r.success))
    
    return results


def get_best_color_profile(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True,
    validate_cached: bool = True,
) -> Optional[Tuple[int, int, int]]:
    """
    Test color profiles and return the best available configuration.
    
    The "best" profile is the first successful one in the provided list,
    which should be ordered from highest to lowest quality/bandwidth.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_COLOR_PROFILES.
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
                validation = verify_color_profiles(
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
    results = verify_color_profiles(profiles=profiles, verbose=verbose)
    
    working = [r.config_tuple for r in results if r.success]
    
    # Save working profiles
    if working:
        serial = get_camera_serial()
        if serial:
            save_working_profiles(serial, working)
            if verbose:
                print(f"\nSaved {len(working)} working profiles for camera {serial}")
    
    return working[0] if working else None


def get_working_color_profiles(
    profiles: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = False,
    use_cache: bool = True
) -> List[Tuple[int, int, int]]:
    """
    Test color profiles and return all working configurations.
    
    Args:
        profiles: List of (width, height, fps) tuples to test.
                 If None, uses DEFAULT_COLOR_PROFILES.
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
                validation = verify_color_profiles(
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
    results = verify_color_profiles(profiles=profiles, verbose=verbose)
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
    print("RealSense Color Profile Verification")
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
        results = verify_color_profiles(verbose=True)
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