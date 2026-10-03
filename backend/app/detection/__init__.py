"""
detection package
-----------------
Wraps the trained YOLO11s model and exposes a clean, typed interface for
performing object detection on sonar images.

Public symbols
~~~~~~~~~~~~~~
- :class:`Detection`   – single bounding-box result
- :class:`YOLODetector` – thin, stateful wrapper around the trained weights
"""

from .yolo_detector import Detection, YOLODetector

__all__ = ["Detection", "YOLODetector"]
