import time, pyrealsense2 as rs

ctx = rs.context()
print("Devices:", [d.get_info(rs.camera_info.name) for d in ctx.devices])
assert len(ctx.devices) > 0, "no device"

dev = ctx.devices[0]
# Enable emitter if present
try:
    stereo = next(s for s in dev.query_sensors() if "Stereo" in s.get_info(rs.camera_info.name))
    if stereo.supports(rs.option.emitter_enabled):
        stereo.set_option(rs.option.emitter_enabled, 1)
        print("Emitter enabled")
except Exception as e:
    print("Emitter setup skipped:", e)

cfg = rs.config()
# Conservative depth stream; we’ll relax fps and resolution
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

pipe = rs.pipeline()
for attempt in range(1, 6):
    try:
        print(f"Starting pipeline (attempt {attempt})...")
        pipe.start(cfg)
        break
    except Exception as e:
        print("Start failed:", e)
        time.sleep(0.5)
else:
    raise SystemExit("Could not start depth")

ok = False
for i in range(20):
    fs = pipe.wait_for_frames(2000)
    d = fs.get_depth_frame()
    if d:
        print("Depth frame:", d.get_width(), "x", d.get_height())
        ok = True
        break
if not ok:
    raise SystemExit("No depth frame within 20 tries")
pipe.stop()
print("Depth OK")