"""
Inference helper for YOLOv11 (.pt) object-detection models using PyTorch/Ultralytics.

Designed to be imported by other Python programs. It exposes two main APIs:

1) load_model(model_path: str, device: Optional[str] = None) -> YOLO
2) infer(
       model: "YOLO",
       image: Union[str, "PIL.Image.Image", "np.ndarray", "torch.Tensor"],
       conf: float = 0.25,
       iou: float = 0.45,
       max_det: int = 300,
   ) -> Tuple[List["Detection"], List[Tuple[float, float, float, float]]]

The `infer` function now supports `bbox_format` ("xyxy" | "xywh" | "cxcywh") and `normalized` (bool) parameters.
A new `infer_video_stream` generator is available for continuous sources (files, webcams, RTSP).

Returns:
- detections: list of Detection dataclass instances sorted by confidence (desc)
- bboxes_xyxy: list of (x1, y1, x2, y2) tuples in the same sorted order

This module depends on `torch` and `ultralytics` (which provides YOLOv11
runtimes on top of PyTorch). Install with:

    pip install ultralytics torch --upgrade

Example
-------
>>> from pickafresa_vision.inference_node_bbox import load_model, infer, detections_to_dicts, infer_video_stream
>>> model = load_model("/path/to/your/yolo11.pt")
>>> detections, bboxes = infer(model, "/path/to/image.jpg", conf=0.3)
>>> print(bboxes)  # [(x1, y1, x2, y2), ...] sorted by confidence
>>> print(detections_to_dicts(detections))  # convenient JSON-serializable format

>>> # Using bbox_format and normalized parameters
>>> detections, bboxes = infer(model, "/path/to/image.jpg", conf=0.3, bbox_format='xywh', normalized=True)
>>> print(bboxes)  # bounding boxes in xywh format normalized to [0,1]

>>> # Using infer_video_stream for webcam or video file
>>> for frame_idx, dets, bboxes, frame in infer_video_stream(model, 0, conf=0.3, bbox_format='cxcywh'):
>>>     print(f"Frame {frame_idx}: {len(dets)} detections")

>>> # Using display=True to show results visually for a single image
>>> detections, bboxes = infer(model, "/path/to/image.jpg", conf=0.3, display=True)

>>> # Using display=True with infer_video_stream to show video with detections
>>> for frame_idx, dets, bboxes, frame in infer_video_stream(model, 0, conf=0.3, display=True):
>>>     pass

# @aldrick-t, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Union, Generator

# Optional imports for type hints only (to avoid hard runtime deps on these names)
try:  # pragma: no cover - for typing convenience
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = object  # type: ignore

try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = object  # type: ignore

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

import torch


@dataclass(frozen=True)
class Detection:
    """Container for a single model prediction.

    Attributes
    ----------
    clazz : str
        Human-readable class name (e.g., "person").
    class_id : int
        Integer class index as predicted by the model.
    confidence : float
        Confidence score in the range [0, 1].
    bbox_xyxy : Tuple[float, float, float, float]
        Bounding box in absolute pixel coordinates (x1, y1, x2, y2).
    """

    clazz: str
    class_id: int
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert corner format to width/height center-origin format (top-left x,y, width, height)."""
    return (x1, y1, x2 - x1, y2 - y1)

def _xyxy_to_cxcywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert to center x/y plus width/height."""
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return (cx, cy, w, h)

def _normalize_bbox(bb: Tuple[float, float, float, float], fmt: str, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Normalize bbox coordinates to [0,1] according to the format."""
    x, y, w, h = bb if fmt != "xyxy" else _xyxy_to_xywh(*bb)
    return (x / img_w, y / img_h, w / img_w, h / img_h)

def _convert_bbox(
    bb_xyxy: Tuple[float, float, float, float],
    fmt: str,
    normalized: bool,
    img_w: Optional[int],
    img_h: Optional[int],
) -> Tuple[float, float, float, float]:
    """Convert an (x1,y1,x2,y2) bbox to requested format and optionally normalize."""
    x1, y1, x2, y2 = bb_xyxy
    if fmt == "xyxy":
        bb = (x1, y1, x2, y2)
    elif fmt == "xywh":
        bb = _xyxy_to_xywh(x1, y1, x2, y2)
    elif fmt == "cxcywh":
        bb = _xyxy_to_cxcywh(x1, y1, x2, y2)
    else:
        raise ValueError(f"Unsupported bbox_format: {fmt}. Use 'xyxy', 'xywh', or 'cxcywh'.")

    if normalized:
        if img_w is None or img_h is None:
            raise ValueError("Normalization requested but image size is unknown.")
        # Normalize in xywh space for consistency
        if fmt == "xyxy":
            bb_xywh = _xyxy_to_xywh(x1, y1, x2, y2)
            nx, ny, nw, nh = _normalize_bbox(bb_xywh, "xywh", img_w, img_h)
            # convert back to xyxy normalized
            return (nx, ny, nx + nw, ny + nh)
        else:
            nx, ny, nw, nh = _normalize_bbox(bb if fmt != "xyxy" else _xyxy_to_xywh(*bb), fmt, img_w, img_h)
            if fmt == "xywh":
                return (nx, ny, nw, nh)
            # cxcywh
            return (nx, ny, nw, nh)
    return bb


def _draw_and_optionally_show(
    frame: "np.ndarray",
    dets: List[Detection],
    bboxes: List[Tuple[float, float, float, float]],
    bbox_format: str,
    normalized: bool,
    class_labels: bool = True,
    window_name: str = "YOLOv11 Inference",
    show: bool = False,
    delay: int = 1,
) -> None:
    """Draw bboxes (and optional labels) on frame; optionally show via cv2.imshow()."""
    if cv2 is None:
        if show:
            raise RuntimeError("OpenCV required for display. Install with `pip install opencv-python`.")
        return
    h, w = frame.shape[:2]
    # Convert all bboxes to pixel xyxy for drawing
    def to_xyxy(bb):
        if bbox_format == "xyxy" and not normalized:
            return (bb[0], bb[1], bb[2], bb[3])
        # if normalized, convert to pixels first
        if normalized:
            if bbox_format == "xyxy":
                x1 = bb[0] * w; y1 = bb[1] * h; x2 = bb[2] * w; y2 = bb[3] * h
                return (x1, y1, x2, y2)
            elif bbox_format == "xywh":
                x = bb[0] * w; y = bb[1] * h; bw = bb[2] * w; bh = bb[3] * h
                return (x, y, x + bw, y + bh)
            else:  # cxcywh
                cx = bb[0] * w; cy = bb[1] * h; bw = bb[2] * w; bh = bb[3] * h
                return (cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2)
        else:
            if bbox_format == "xyxy":
              return (bb[0], bb[1], bb[2], bb[3])
            elif bbox_format == "xywh":
              return (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
            else:  # cxcywh
              return (bb[0] - bb[2]/2, bb[1] - bb[3]/2, bb[0] + bb[2]/2, bb[1] + bb[3]/2)
    for det, bb in zip(dets, bboxes):
        x1, y1, x2, y2 = to_xyxy(bb)
        pt1 = (int(max(0, min(w-1, x1))), int(max(0, min(h-1, y1))))
        pt2 = (int(max(0, min(w-1, x2))), int(max(0, min(h-1, y2))))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        if class_labels:
            label = f"{det.clazz} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (pt1[0], pt1[1] - th - 6), (pt1[0] + tw + 2, pt1[1]), (0, 255, 0), -1)
            cv2.putText(frame, label, (pt1[0] + 1, pt1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if show:
        cv2.imshow(window_name, frame)
        cv2.waitKey(delay)


def load_model(model_path: str, device: Optional[str] = None):
    """Load a YOLOv11 model (.pt) using Ultralytics on the requested device.

    Parameters
    ----------
    model_path : str
        Path to the YOLOv11 .pt weights file.
    device : Optional[str]
        e.g., "cuda", "cpu", or "mps". If None, auto-selects.

    Returns
    -------
    model : YOLO
        A ready-to-run Ultralytics YOLO model moved to the target device.
    """
    dev = _select_device(device)
    try:
        from ultralytics import YOLO  # lazy import
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is required. Install with `pip install ultralytics`."
        ) from e

    model = YOLO(model_path)
    # Move the model to the target device when applicable
    try:
        model.to(dev)
    except Exception:
        # Some backends handle device internally; ignore if not supported
        pass
    return model


def _select_device(device: Optional[str] = None) -> str:
    """Choose a compute device string.

    If `device` is None, prefer CUDA if available, otherwise CPU.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def infer(
    model,
    image: Union[str, "Image", "np.ndarray", torch.Tensor],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    bbox_format: str = "xyxy",
    normalized: bool = False,
    display: bool = False,
    display_window: str = "YOLOv11 Inference",
    display_labels: bool = True,
) -> Tuple[List[Detection], List[Tuple[float, float, float, float]]]:
    """Run inference on a single image-like input.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model as returned by `load_model`.
    image : Union[str, PIL.Image.Image, np.ndarray, torch.Tensor]
        Any image format accepted by Ultralytics (path, PIL, ndarray, tensor).
    conf : float, default 0.25
        Confidence threshold.
    iou : float, default 0.45
        IOU threshold for NMS.
    max_det : int, default 300
        Maximum number of detections per image.
    bbox_format : str, default "xyxy"
        Format of bounding boxes to return: "xyxy", "xywh", or "cxcywh".
    normalized : bool, default False
        Whether to normalize bounding box coordinates to [0,1].
    display : bool, default False
        Whether to display the image with detections using OpenCV.
    display_window : str, default "YOLOv11 Inference"
        Window name for display.
    display_labels : bool, default True
        Whether to show class labels on the displayed image.

    Returns
    -------
    detections : List[Detection]
        List of predictions sorted by confidence (descending).
    bboxes_xyxy : List[Tuple[float, float, float, float]]
        Bounding boxes sorted in the same order, converted to requested format.
    """
    # Run the model
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return [], []

    r = results[0]
    h, w = getattr(r, "orig_shape", (None, None))
    if (h is None or w is None) and hasattr(r, "orig_img") and r.orig_img is not None:
        try:
            h, w = r.orig_img.shape[:2]
        except Exception:
            h, w = None, None

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []

    # Extract tensors
    xyxy = boxes.xyxy.detach().cpu().tolist()  # list[list[float]] (N,4)
    confs = boxes.conf.detach().cpu().tolist()  # list[float] (N)
    cls_ids = boxes.cls.detach().cpu().to(torch.int64).tolist()  # list[int] (N)

    # Class name mapping
    names = getattr(r, "names", None) or getattr(model, "names", None) or {}

    dets: List[Detection] = []
    for cid, score, bb in zip(cls_ids, confs, xyxy):
        clazz = names.get(int(cid), str(cid)) if isinstance(names, dict) else str(cid)
        dets.append(
            Detection(
                clazz=clazz,
                class_id=int(cid),
                confidence=float(score),
                bbox_xyxy=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
            )
        )

    # Sort by confidence descending
    dets.sort(key=lambda d: d.confidence, reverse=True)
    bboxes_sorted = [
        _convert_bbox(d.bbox_xyxy, bbox_format, normalized, w, h)
        for d in dets
    ]

    if display:
        frame = None
        # Try to get a frame to draw on
        if hasattr(r, "orig_img") and r.orig_img is not None:
            frame = r.orig_img.copy()
        elif isinstance(image, str) and cv2 is not None:
            frame = cv2.imread(image)
        if frame is not None:
            _draw_and_optionally_show(
                frame,
                dets,
                bboxes_sorted,
                bbox_format,
                normalized,
                class_labels=display_labels,
                window_name=display_window,
                show=True,
                delay=0,
            )
            if cv2 is not None:
                cv2.waitKey(0)
                cv2.destroyWindow(display_window)

    return dets, bboxes_sorted


def detections_to_dicts(detections: List[Detection]) -> List[dict]:
    """Convert Detection objects to plain dicts (JSON-serializable)."""
    return [asdict(d) for d in detections]


def infer_video_stream(
    model,
    source: Union[int, str],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    bbox_format: str = "xyxy",
    normalized: bool = False,
    read_fps: Optional[float] = None,
    return_frames: bool = False,
    display: bool = False,
    display_window: str = "YOLOv11 Inference",
    display_labels: bool = True,
    quit_key: str = 'q',
) -> Generator[Tuple[int, List[Detection], List[Tuple[float, float, float, float]], Optional["np.ndarray"]], None, None]:
    """
    Iterate over a continuous video source (file path, webcam index, RTSP/HTTP).

    Yields tuples of (frame_idx, detections, bboxes, frame_or_None), with
    bboxes ordered by confidence and formatted/normalized as requested.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model as returned by `load_model`.
    source : Union[int, str]
        Video source (webcam index, file path, or stream URL).
    conf : float, default 0.25
        Confidence threshold.
    iou : float, default 0.45
        IOU threshold for NMS.
    max_det : int, default 300
        Maximum number of detections per frame.
    bbox_format : str, default "xyxy"
        Format of bounding boxes to return: "xyxy", "xywh", or "cxcywh".
    normalized : bool, default False
        Whether to normalize bounding box coordinates to [0,1].
    read_fps : Optional[float], default None
        Optional FPS to read frames at.
    return_frames : bool, default False
        Whether to yield the frame along with detections.
    display : bool, default False
        Whether to display frames with detections using OpenCV.
    display_window : str, default "YOLOv11 Inference"
        Window name for display.
    display_labels : bool, default True
        Whether to show class labels on the displayed frames.
    quit_key : str, default 'q'
        Key to press to quit the display window.

    Yields
    ------
    Tuple[int, List[Detection], List[Tuple[float, float, float, float]], Optional[np.ndarray]]
        Frame index, list of detections, list of bounding boxes in requested format,
        and optionally the frame itself.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for video streaming. Install with `pip install opencv-python`." )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    if read_fps is None:
        read_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Run inference per frame
            results = model.predict(
                frame,
                conf=conf,
                iou=iou,
                max_det=max_det,
                verbose=False,
            )
            if not results:
                yield frame_idx, [], [], (frame if return_frames else None)
                frame_idx += 1
                continue

            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                yield frame_idx, [], [], (frame if return_frames else None)
                frame_idx += 1
                continue

            h, w = getattr(r, "orig_shape", frame.shape[:2])

            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            cls_ids = boxes.cls.detach().cpu().to(torch.int64).tolist()
            names = getattr(r, "names", None) or getattr(model, "names", None) or {}

            dets: List[Detection] = []
            for cid, score, bb in zip(cls_ids, confs, xyxy):
                clazz = names.get(int(cid), str(cid)) if isinstance(names, dict) else str(cid)
                dets.append(
                    Detection(
                        clazz=clazz,
                        class_id=int(cid),
                        confidence=float(score),
                        bbox_xyxy=(float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])),
                    )
                )

            dets.sort(key=lambda d: d.confidence, reverse=True)
            bboxes_sorted = [
                _convert_bbox(d.bbox_xyxy, bbox_format, normalized, int(w), int(h))
                for d in dets
            ]

            if display:
                _draw_and_optionally_show(
                    frame,
                    dets,
                    bboxes_sorted,
                    bbox_format,
                    normalized,
                    class_labels=display_labels,
                    window_name=display_window,
                    show=True,
                    delay=1,
                )
                key = cv2.waitKey(1) & 0xFF
                if key == ord(quit_key):
                    break

            yield frame_idx, dets, bboxes_sorted, (frame if return_frames else None)
            frame_idx += 1
    finally:
        cap.release()
        if display and cv2 is not None:
            cv2.destroyWindow(display_window)


__all__ = [
    "Detection",
    "load_model",
    "infer",
    "detections_to_dicts",
    "infer_video_stream",
]