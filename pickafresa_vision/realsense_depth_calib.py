import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

def start_safe(pipe):
    cfg = rs.config()
    # Try depth first, conservative
    tried = [
        (rs.stream.depth, 640, 480, rs.format.z16, 15),
        (rs.stream.depth, 848, 480, rs.format.z16, 30),
        (rs.stream.depth, 424, 240, rs.format.z16, 30),
    ]
    for st,w,h,fmt,fps in tried:
        c2 = rs.config()
        c2.enable_stream(st, w, h, fmt, fps)
        try:
            return pipe.start(c2)
        except Exception:
            pass
    raise RuntimeError("No viable depth profile found")

# ------------------ CONFIGURATION ------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align = rs.align(rs.stream.color)

print(f"Depth Scale: {depth_scale:.6f} m/unit")

# ------------------ VARIABLES ------------------
clicked_point = None
current_depth = None
real_distance = ""
measurements = []
show_text = "Click to get depth measurement. Write actual distance."

def get_depth_value(depth_frame, x, y, window=5):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    half = window // 2

    vals = []
    for yy in range(max(0, y - half), min(h, y + half + 1)):
        for xx in range(max(0, x - half), min(w, x + half + 1)):
            d = depth_frame.get_distance(xx, yy)
            if d > 0:
                vals.append(d)

    return np.mean(vals) if vals else None


def mouse_callback(event, x, y, flags, param):
    global clicked_point, current_depth, show_text
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame = param["depth_frame"]
        dist = get_depth_value(depth_frame, x, y)
        if dist:
            clicked_point = (x, y)
            current_depth = dist
            show_text = f"Point ({x},{y}) = {dist:.3f} m → write actual distance"
        else:
            show_text = "Valid distance could not be obtained."

cv2.namedWindow("RealSense Calibration")
callback_params = {}
cv2.setMouseCallback("RealSense Calibration", mouse_callback, callback_params)

print("\n Interactive RealSense D435 Depth Calibration")
print("1) Click on a point to measure distance.")
print("2) Type the actual distance (m) directly on the keyboard.")
print("3) Press Enter to save the point.")
print("4) Press 'q' to exit.\n")

try:
    while True:
        frames = pipeline.wait_for_frames(10000)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        callback_params["depth_frame"] = depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        combined = cv2.addWeighted(color_image, 0.6, depth_colormap, 0.4, 0)

        # Draw current clicked point and depth
        if clicked_point:
            cv2.circle(combined, clicked_point, 5, (0, 255, 255), -1)
            if current_depth:
                cv2.putText(combined, f"{current_depth:.3f} m", 
                            (clicked_point[0]+10, clicked_point[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.putText(combined, show_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if real_distance:
            cv2.putText(combined, f"Actual distance: {real_distance}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("RealSense Calibration", combined)
        key = cv2.waitKey(1) & 0xFF

    
        if key == ord('q'):
            break

        # If Enter is pressed
        elif key == 13 and current_depth and real_distance:
            try:
                val = float(real_distance)
                measurements.append((val, current_depth))
                print(f"Saved: real={val:.3f} m, camera={current_depth:.3f} m\n")
                show_text = "Saved point. Click on another point."
                real_distance = ""
                current_depth = None
                clicked_point = None
            except ValueError:
                show_text = "Invalid entry!"
                real_distance = ""

        # If Backspace is pressed
        elif key == 8:
            real_distance = real_distance[:-1]

       # If numbers or a period are pressed
        elif 48 <= key <= 57 or key == ord('.'):
            real_distance += chr(key)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ------------------ Calibration ------------------
if len(measurements) < 2:
    print("Not enough points to calibrate!")
    sys.exit()

df = pd.DataFrame(measurements, columns=["real", "cam"])
df.to_csv("realsense_calib.csv", index=False)
print("\n Data saved in 'realsense_calib.csv''.")

X = df["cam"].values.reshape(-1, 1)
y = df["real"].values
model = LinearRegression().fit(X, y)
a, b = model.coef_[0], model.intercept_

df["pred"] = model.predict(X)
df["error_mm"] = (df["pred"] - df["real"]) * 1000
error_prom = abs(df["error_mm"]).mean()

print(f"\n Fitted model:")
print(f"Z_real = {a:.4f} * Z_cam + {b:.4f}")
print(f"Average error: {error_prom:.1f} mm\n")

plt.figure(figsize=(6, 5))
plt.scatter(df["cam"], df["real"], label="Measurements", color="orange")
plt.plot(df["cam"], df["pred"], color="blue", label=f"Ajuste: Zr={a:.3f}*Zc+{b:.3f}")
plt.xlabel("Camera distance (m)")
plt.ylabel("Actual distance (m)")
plt.title("RealSense D435 Depth Calibration")
plt.legend()
plt.grid(True)
plt.show()
