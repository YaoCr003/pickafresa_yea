'''
RealSense D400 Series Color Stream Verification Tool
Tests if the color stream can be started with a basic configuration.

Team YEA, 2025
'''

import pyrealsense2 as rs
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)
pipe.stop()
print("Color stream OK")