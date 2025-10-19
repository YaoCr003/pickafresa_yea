import time, pyrealsense2 as rs

# Depth profile sweep for D435: test multiple modes from high→low bandwidth
# and report which ones produce frames.

ctx = rs.context()
print("Devices:", [d.get_info(rs.camera_info.name) for d in ctx.devices])
assert len(ctx.devices) > 0, "no device"

dev = ctx.devices[0]

# Try to enable the emitter when available (helps depth on some units)
try:
    stereo = next(s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name))
    if stereo.supports(rs.option.emitter_enabled):
        stereo.set_option(rs.option.emitter_enabled, 1)
        print("Emitter enabled")
except Exception as e:
    print("Emitter setup skipped:", e)

# List of (width, height, fps) to try. Ordered high→low bandwidth.
ATTEMPTS = [
    (1280, 720, 30),
    (1280, 720, 15),
    (848, 480, 30),
    (848, 480, 15),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
    (424, 240, 15),
    (320, 240,  6),
]

results = []

for (w, h, fps) in ATTEMPTS:
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    pipe = rs.pipeline()
    print(f"\n=== Testing depth {w}x{h}@{fps} ===")
    started = False
    try:
        try:
            pipe.start(cfg)
            started = True
        except Exception as e:
            print("Start failed:", e)
            results.append(((w, h, fps), False, f"start: {e}"))
            continue

        ok = False
        first_exc = None
        for i in range(10):  # up to ~15s total depending on wait
            try:
                frames = pipe.wait_for_frames(1500)
                d = frames.get_depth_frame() if frames else None
                if d:
                    print("PASS: frame", d.get_width(), "x", d.get_height())
                    ok = True
                    break
            except Exception as e:
                if first_exc is None:
                    first_exc = e
                time.sleep(0.2)
        if ok:
            results.append(((w, h, fps), True, "ok"))
        else:
            msg = f"no frame within timeout; first_exc={first_exc}"
            print("FAIL:", msg)
            results.append(((w, h, fps), False, msg))
    finally:
        if started:
            try:
                pipe.stop()
                time.sleep(0.2)
            except Exception:
                pass

# Summary
print("\n===== SUMMARY =====")
for (w, h, fps), ok, note in results:
    print(f"{w}x{h}@{fps}: {'OK' if ok else 'FAIL'}  -- {note}")

# Exit non-zero if none worked (useful in CI)
if not any(ok for (_, ok, _) in [(p, ok, note) for (p, ok, note) in results]):
    raise SystemExit(1)
else:
    print("\nAt least one depth profile succeeded.")