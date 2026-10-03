"""
yolo_detector.py
----------------
Thin, stateful wrapper around the trained Drishti YOLO11s model.

Responsibilities
~~~~~~~~~~~~~~~~
- Load ``best.pt`` exactly once at construction time.
- Accept a single sonar image (NumPy array or file path).
- Run inference at a configurable confidence threshold.
- Return a typed list of :class:`Detection` objects — one per bounding box.

Nothing here retrains, fine-tunes, or modifies the model weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical class names, indexed by YOLO class-id.
CLASS_NAMES: dict[int, str] = {
    0: "submarine_pipeline",
    1: "shipwreck",
    2: "ghost_net",
    3: "mine_cylinder",
}

#: Absolute path to the trained weights used by default.
DEFAULT_WEIGHTS: Path = Path(
    r"C:\Users\Sufiyan Ahamed\Desktop\drishti-sss"
    r"\runs\detect\runs\drishti_baseline-3\weights\best.pt"
)

#: Default confidence threshold (matches YOLO's own default of 0.25).
DEFAULT_CONF: float = 0.25


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Detection:
    """
    A single object detection returned by :class:`YOLODetector`.

    Attributes
    ----------
    class_id:
        Integer class index as produced by the model (0 … 3).
    class_name:
        Human-readable class label mapped from *class_id*.
    confidence:
        Detection confidence score in [0.0, 1.0].
    x1, y1:
        Top-left corner of the bounding box in pixel coordinates.
    x2, y2:
        Bottom-right corner of the bounding box in pixel coordinates.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)`` as a plain tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class YOLODetector:
    """
    Wrapper around the trained YOLO11s Drishti model.

    Parameters
    ----------
    weights_path:
        Path to a ``*.pt`` weights file.  Defaults to the trained
        ``best.pt`` produced during drishti_baseline-3 training.
    conf_threshold:
        Minimum confidence for a detection to be included in the results.
        Must be in (0, 1].  Defaults to 0.25.

    Raises
    ------
    FileNotFoundError
        If *weights_path* does not exist on disk.
    ValueError
        If *conf_threshold* is outside (0, 1].
    """

    def __init__(
        self,
        weights_path: Union[str, Path, None] = None,
        conf_threshold: float = DEFAULT_CONF,
    ) -> None:
        # Resolve weights path
        resolved = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        if not resolved.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {resolved}"
            )

        # Validate confidence threshold
        if not (0.0 < conf_threshold <= 1.0):
            raise ValueError(
                f"conf_threshold must be in (0, 1], got {conf_threshold}"
            )

        self._weights_path: Path = resolved
        self._conf_threshold: float = conf_threshold

        # Lazy-load the YOLO model (import deferred so the module can be
        # imported without ultralytics installed in test environments that
        # mock it out).
        from ultralytics import YOLO  # noqa: PLC0415

        self._model = YOLO(str(self._weights_path))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def conf_threshold(self) -> float:
        """Current confidence threshold."""
        return self._conf_threshold

    @property
    def weights_path(self) -> Path:
        """Resolved path to the loaded weights file."""
        return self._weights_path

    @property
    def class_names(self) -> dict[int, str]:
        """
        Class-id → name mapping sourced from the loaded model.

        Falls back to the module-level :data:`CLASS_NAMES` constant if the
        model does not expose ``names``.
        """
        model_names: dict[int, str] | None = getattr(self._model, "names", None)
        return model_names if model_names else CLASS_NAMES

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[np.ndarray, str, Path],
    ) -> list[Detection]:
        """
        Run inference on a single sonar image.

        Parameters
        ----------
        image:
            Either a NumPy ``uint8`` BGR/grayscale array (H × W × C or H × W),
            or a file path (``str`` / :class:`~pathlib.Path`) to an image on
            disk.

        Returns
        -------
        list[Detection]
            Zero or more :class:`Detection` objects, one per bounding box
            that exceeds the configured confidence threshold.

        Raises
        ------
        TypeError
            If *image* is not a NumPy array or a path-like string.
        ValueError
            If a NumPy array is provided but has an unsupported shape or dtype.
        FileNotFoundError
            If a path is provided but does not point to an existing file.
        """
        image = self._validate_image(image)

        results = self._model(
            image,
            conf=self._conf_threshold,
            verbose=False,
        )

        return self._parse_results(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(
        image: Union[np.ndarray, str, Path],
    ) -> Union[np.ndarray, str]:
        """Validate and normalise the *image* argument."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(
                    f"Image file not found: {path}"
                )
            return str(path)

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a NumPy array or a file path, "
                f"got {type(image).__name__!r}"
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                f"NumPy image must be 2-D (grayscale) or 3-D (H×W×C), "
                f"got ndim={image.ndim}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"NumPy image dtype must be uint8, got {image.dtype}"
            )

        return image

    def _parse_results(self, results: list) -> list[Detection]:
        """
        Convert the raw ultralytics ``Results`` list into :class:`Detection`
        objects.

        Only the first element of *results* is used (single-image inference).
        """
        if not results:
            return []

        result = results[0]
        boxes = result.boxes

        # No detections at all
        if boxes is None or len(boxes) == 0:
            return []

        names = self.class_names
        detections: list[Detection] = []

        xyxy = boxes.xyxy.cpu().numpy()      # shape (N, 4) – pixel coords
        confs = boxes.conf.cpu().numpy()      # shape (N,)
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=names.get(cid, f"class_{cid}"),
                    confidence=float(confs[i]),
                    x1=float(xyxy[i, 0]),
                    y1=float(xyxy[i, 1]),
                    x2=float(xyxy[i, 2]),
                    y2=float(xyxy[i, 3]),
                )
            )

        return detections
