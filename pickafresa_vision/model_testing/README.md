# Object Detection Model Testing Tools

This directory contains tools for testing and benchmarking YOLO object detection models.

## `objd_testing.py` - Real-time Video Testing Tool

Interactive tool for testing object detection models on live video feeds with visual feedback.

### Features

- **Model Selection**: Choose from available `.pt` model files
- **Dataset Integration**: Load class names from dataset `data.yaml` files
- **Multi-camera Support**:
  - RealSense D400 series cameras (with depth)
  - Standard USB/built-in cameras via OpenCV
- **Visual Overlays**:
  - Class-specific bounding box colors (Green=Ripe, Yellow=Unripe, Blue=Flower)
  - Confidence scores
  - Depth measurements (RealSense only)
  - Real-time statistics (FPS, detection counts)
- **Interactive Controls**:
  - Adjust confidence threshold on-the-fly
  - Pause/resume inference
  - Save frames
  - Reset settings
- **Configuration Persistence**: Remembers your last settings

### Usage

```bash
# Standard Python
python pickafresa_vision/model_testing/objd_testing.py

# Or with RealSense venv wrapper (if using RealSense)
./realsense_venv_sudo pickafresa_vision/model_testing/objd_testing.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save current frame to `pickafresa_vision/images/` |
| `p` | Pause/Resume inference |
| `+` or `=` | Increase confidence threshold by 0.05 |
| `-` or `_` | Decrease confidence threshold by 0.05 |
| `r` | Reset to default settings (conf=0.25, iou=0.45) |

### Interactive Setup

On first run (or when choosing new configuration), you'll be prompted for:

1. **Model Selection**: Choose from `.pt` files in `pickafresa_vision/models/`
2. **Dataset Selection**: Choose dataset folder for class name mapping
3. **Camera Type**: RealSense or OpenCV (USB/built-in)
4. **Camera Index**: For OpenCV cameras (usually 0 for built-in)
5. **Depth Mode**: Enable depth overlay (RealSense only)

### Display Layout

**Without Depth Overlay:**
```
┌────────────────────────────────────┐
│ Stats Overlay (upper left)         │
│ - Model name                       │
│ - Camera info                      │
│ - FPS                              │
│ - Detection counts                 │
│                                    │
│    [Color frame with bboxes]       │
│                                    │
│ Controls (bottom)                  │
└────────────────────────────────────┘
```

**With Depth Overlay (RealSense):**
```
┌──────────────────┬──────────────────┐
│ Color + Bboxes   │ Depth Colormap   │
│ + Stats overlay  │                  │
│                  │                  │
│ Controls (bottom across both)       │
└──────────────────┴──────────────────┘
```

### Configuration Files

Settings are persisted in:
- `~/.pickafresa_vision/config.json` (user-level)
- `pickafresa_vision/.user_config.json` (project-level)

### Dependencies

- `ultralytics` - YOLO model inference
- `opencv-python` - Video capture and display
- `numpy` - Array operations
- `pyrealsense2` - RealSense camera support (optional)
- `pyyaml` - Dataset configuration parsing (optional)

### Class Colors

Default color coding for bounding boxes:
- **Ripe**: Green (0, 255, 0)
- **Unripe**: Yellow (0, 255, 255)
- **Flower**: Blue (255, 0, 0)

Colors are in BGR format for OpenCV compatibility.

### Depth Measurements

When depth overlay is enabled with RealSense cameras:
- Depth values are calculated as the median depth within each bounding box ROI
- Displayed in millimeters (mm) next to confidence scores
- Depth filtering pipeline applied for improved accuracy:
  - Disparity transform
  - Spatial filtering
  - Temporal filtering
  - Hole filling

### Troubleshooting

**"pyrealsense2 not available"**
- Install RealSense SDK: `pip install pyrealsense2`
- Or use OpenCV cameras instead

**"Failed to open camera"**
- Check camera index (try 0, 1, 2...)
- Ensure camera is not in use by another application
- For RealSense: verify with `rs-enumerate-devices`

**Low FPS**
- Try lower resolution (modify `TestConfig.resolution`)
- Reduce confidence threshold to skip more detections
- Disable depth overlay
- Use GPU acceleration (ensure CUDA is available)

**Incorrect class names/colors**
- Verify `data.yaml` exists in selected dataset folder
- Check class names match: "flower", "ripe", "unripe"

### Performance Tips

1. **GPU Acceleration**: Models automatically use CUDA if available
2. **Resolution**: Lower resolution = faster inference
3. **Confidence Threshold**: Higher threshold = fewer detections = faster
4. **Depth Filtering**: Slight performance impact but better depth quality

---

Team YEA, 2025
