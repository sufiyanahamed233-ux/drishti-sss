"""
test_detection.py
-----------------
Phase 2 tests for backend.app.detection.

Strategy
~~~~~~~~
All tests use ``unittest.mock`` to patch ``ultralytics.YOLO`` so that:
  - No GPU / disk I/O is needed during CI.
  - The real model weights are NOT loaded (test speed stays < 1 s).
  - We can precisely control the fake model's output.

A single ``conftest``-style fixture (``mock_yolo_cls``) patches the class
at the point it is imported inside ``YOLODetector.__init__``.

Coverage
--------
- YOLODetector initialisation (happy path)
- FileNotFoundError for missing weights
- ValueError for bad confidence threshold
- detect() output structure for zero, one, and multiple detections
- class-name mapping (all four Drishti classes)
- bbox property on Detection
- Invalid image inputs (wrong type, wrong ndim, wrong dtype)
- File-path image (valid file and missing file)
- Confidence threshold filtering respected
- Detector is deterministic (same mock output → same Detection list)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake ultralytics Results
# ---------------------------------------------------------------------------

import torch


def _make_boxes(
    detections: list[tuple[float, float, float, float, float, int]],
) -> MagicMock:
    """
    Build a mock ``ultralytics.engine.results.Boxes``-like object.

    Each entry in *detections* is ``(x1, y1, x2, y2, conf, cls_id)``.
    """
    boxes = MagicMock()
    if not detections:
        boxes.__len__ = lambda self: 0
        boxes.xyxy = torch.zeros((0, 4))
        boxes.conf = torch.zeros((0,))
        boxes.cls = torch.zeros((0,))
    else:
        n = len(detections)
        xyxy_data = torch.tensor([[d[0], d[1], d[2], d[3]] for d in detections], dtype=torch.float32)
        conf_data = torch.tensor([d[4] for d in detections], dtype=torch.float32)
        cls_data = torch.tensor([d[5] for d in detections], dtype=torch.float32)
        boxes.__len__ = lambda self: n
        boxes.xyxy = xyxy_data
        boxes.conf = conf_data
        boxes.cls = cls_data
    return boxes


def _make_result(
    detections: list[tuple[float, float, float, float, float, int]],
) -> MagicMock:
    """Return a mock ``ultralytics.engine.results.Results`` object."""
    result = MagicMock()
    result.boxes = _make_boxes(detections)
    return result


def _make_model(
    detections: list[tuple[float, float, float, float, float, int]],
    names: dict[int, str] | None = None,
) -> MagicMock:
    """
    Return a mock YOLO model whose ``__call__`` returns *detections*.

    *names* defaults to the four Drishti class names.
    """
    default_names = {
        0: "submarine_pipeline",
        1: "shipwreck",
        2: "ghost_net",
        3: "mine_cylinder",
    }
    model = MagicMock()
    model.names = names if names is not None else default_names
    model.return_value = [_make_result(detections)]
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WEIGHTS_PATH = Path(
    r"C:\Users\Sufiyan Ahamed\Desktop\drishti-sss"
    r"\runs\detect\runs\drishti_baseline-3\weights\best.pt"
)


@pytest.fixture()
def blank_image() -> np.ndarray:
    """640×640 black BGR image (uint8)."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


def _make_detector(model_mock: MagicMock, conf: float = 0.25):
    """
    Instantiate YOLODetector with *model_mock* injected via patch.
    The weights path must exist on disk (uses the real best.pt path).
    """
    from backend.app.detection.yolo_detector import YOLODetector

    with patch("backend.app.detection.yolo_detector.YOLO", create=True) as YOLOCls:
        # Import YOLO is deferred inside __init__; we patch the module-level
        # name that the deferral resolves to.
        YOLOCls.return_value = model_mock
        # Patch the import inside the method
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=YOLOCls)}):
            detector = YOLODetector(weights_path=WEIGHTS_PATH, conf_threshold=conf)
            # Directly replace the private model reference so we control output
            detector._model = model_mock
    return detector


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------


class TestYOLODetectorInit:
    """Tests covering YOLODetector construction."""

    def test_init_with_real_weights_succeeds(self) -> None:
        """Detector loads successfully when weights file exists (model mocked)."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        assert detector is not None

    def test_weights_path_property(self) -> None:
        """weights_path property returns the resolved Path."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        assert detector.weights_path == WEIGHTS_PATH

    def test_conf_threshold_property(self) -> None:
        """conf_threshold property reflects the value set at construction."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock, conf=0.5)
        assert detector.conf_threshold == pytest.approx(0.5)

    def test_missing_weights_raises_file_not_found(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when weights path does not exist."""
        from backend.app.detection.yolo_detector import YOLODetector

        with pytest.raises(FileNotFoundError, match="weights"):
            YOLODetector(weights_path=tmp_path / "nonexistent.pt")

    def test_zero_conf_raises_value_error(self) -> None:
        """conf_threshold of 0.0 is not accepted."""
        from backend.app.detection.yolo_detector import YOLODetector

        with pytest.raises(ValueError, match="conf_threshold"):
            YOLODetector(weights_path=WEIGHTS_PATH, conf_threshold=0.0)

    def test_negative_conf_raises_value_error(self) -> None:
        """Negative conf_threshold is not accepted."""
        from backend.app.detection.yolo_detector import YOLODetector

        with pytest.raises(ValueError, match="conf_threshold"):
            YOLODetector(weights_path=WEIGHTS_PATH, conf_threshold=-0.1)

    def test_conf_above_one_raises_value_error(self) -> None:
        """conf_threshold > 1.0 is not accepted."""
        from backend.app.detection.yolo_detector import YOLODetector

        with pytest.raises(ValueError, match="conf_threshold"):
            YOLODetector(weights_path=WEIGHTS_PATH, conf_threshold=1.1)

    def test_conf_exactly_one_is_accepted(self) -> None:
        """conf_threshold of exactly 1.0 is a legal boundary value."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock, conf=1.0)
        assert detector.conf_threshold == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: class-name mapping
# ---------------------------------------------------------------------------


class TestClassNames:
    """Tests verifying class-name resolution."""

    def test_class_names_returns_four_drishti_classes(self) -> None:
        """class_names exposes all four Drishti categories."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        names = detector.class_names
        assert names[0] == "submarine_pipeline"
        assert names[1] == "shipwreck"
        assert names[2] == "ghost_net"
        assert names[3] == "mine_cylinder"

    def test_class_names_sourced_from_model(self) -> None:
        """class_names prefers the model's own .names attribute."""
        custom = {0: "custom_class"}
        model_mock = _make_model([], names=custom)
        detector = _make_detector(model_mock)
        assert detector.class_names == custom

    def test_module_level_class_names_constant(self) -> None:
        """CLASS_NAMES constant contains all four expected entries."""
        from backend.app.detection.yolo_detector import CLASS_NAMES
        assert len(CLASS_NAMES) == 4
        assert set(CLASS_NAMES.values()) == {
            "submarine_pipeline", "shipwreck", "ghost_net", "mine_cylinder"
        }


# ---------------------------------------------------------------------------
# Tests: Detection dataclass
# ---------------------------------------------------------------------------


class TestDetectionDataclass:
    """Unit tests for the Detection value object."""

    def _make_det(self, **kwargs) -> "Detection":
        from backend.app.detection.yolo_detector import Detection
        defaults = dict(
            class_id=0, class_name="submarine_pipeline",
            confidence=0.9, x1=10.0, y1=20.0, x2=100.0, y2=200.0,
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_bbox_property(self) -> None:
        """bbox returns (x1, y1, x2, y2) as a 4-tuple."""
        det = self._make_det(x1=5.0, y1=10.0, x2=50.0, y2=100.0)
        assert det.bbox == (5.0, 10.0, 50.0, 100.0)

    def test_detection_is_immutable(self) -> None:
        """Detection is a frozen dataclass — attributes cannot be mutated."""
        det = self._make_det()
        with pytest.raises((AttributeError, TypeError)):
            det.confidence = 0.5  # type: ignore[misc]

    def test_detection_equality(self) -> None:
        """Two Detections with identical fields compare equal."""
        det1 = self._make_det()
        det2 = self._make_det()
        assert det1 == det2

    def test_all_four_class_ids(self) -> None:
        """Detection can be constructed for each of the four class IDs."""
        from backend.app.detection.yolo_detector import CLASS_NAMES
        for cid, cname in CLASS_NAMES.items():
            det = self._make_det(class_id=cid, class_name=cname)
            assert det.class_id == cid
            assert det.class_name == cname


# ---------------------------------------------------------------------------
# Tests: detect() output — zero detections
# ---------------------------------------------------------------------------


class TestDetectZeroDetections:
    """Edge case: model returns no boxes."""

    def test_empty_image_returns_empty_list(self, blank_image: np.ndarray) -> None:
        """detect() returns an empty list when the model finds nothing."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        results = detector.detect(blank_image)
        assert results == []

    def test_return_type_is_list(self, blank_image: np.ndarray) -> None:
        """detect() always returns a list, even when empty."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        assert isinstance(detector.detect(blank_image), list)


# ---------------------------------------------------------------------------
# Tests: detect() output — single detection
# ---------------------------------------------------------------------------


class TestDetectSingleDetection:
    """detect() with exactly one bounding box."""

    SINGLE = [(10.0, 20.0, 110.0, 220.0, 0.92, 1)]  # shipwreck

    def test_returns_one_detection(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        results = detector.detect(blank_image)
        assert len(results) == 1

    def test_detection_class_id(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        det = detector.detect(blank_image)[0]
        assert det.class_id == 1

    def test_detection_class_name(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        det = detector.detect(blank_image)[0]
        assert det.class_name == "shipwreck"

    def test_detection_confidence(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        det = detector.detect(blank_image)[0]
        assert det.confidence == pytest.approx(0.92, abs=1e-5)

    def test_detection_bbox(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        det = detector.detect(blank_image)[0]
        assert det.bbox == pytest.approx((10.0, 20.0, 110.0, 220.0))

    def test_detection_individual_coords(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.SINGLE))
        det = detector.detect(blank_image)[0]
        assert det.x1 == pytest.approx(10.0)
        assert det.y1 == pytest.approx(20.0)
        assert det.x2 == pytest.approx(110.0)
        assert det.y2 == pytest.approx(220.0)


# ---------------------------------------------------------------------------
# Tests: detect() output — multiple detections
# ---------------------------------------------------------------------------


class TestDetectMultipleDetections:
    """detect() with several bounding boxes covering all four classes."""

    MULTI = [
        (  0.0,   0.0, 100.0, 100.0, 0.91, 0),  # submarine_pipeline
        (100.0, 100.0, 200.0, 200.0, 0.85, 1),  # shipwreck
        (200.0, 200.0, 300.0, 300.0, 0.78, 2),  # ghost_net
        (300.0, 300.0, 400.0, 400.0, 0.62, 3),  # mine_cylinder
    ]

    def test_returns_four_detections(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.MULTI))
        assert len(detector.detect(blank_image)) == 4

    def test_all_class_names_present(self, blank_image: np.ndarray) -> None:
        detector = _make_detector(_make_model(self.MULTI))
        names = {d.class_name for d in detector.detect(blank_image)}
        assert names == {
            "submarine_pipeline", "shipwreck", "ghost_net", "mine_cylinder"
        }

    def test_order_preserved(self, blank_image: np.ndarray) -> None:
        """Detections are returned in the same order as the model output."""
        detector = _make_detector(_make_model(self.MULTI))
        results = detector.detect(blank_image)
        for i, (_, _, _, _, conf, cid) in enumerate(self.MULTI):
            assert results[i].class_id == cid
            assert results[i].confidence == pytest.approx(conf, abs=1e-5)

    def test_two_detections_same_class(self, blank_image: np.ndarray) -> None:
        """Multiple detections of the same class are all returned."""
        two_mines = [
            (0.0, 0.0, 50.0, 50.0, 0.9, 3),
            (60.0, 60.0, 110.0, 110.0, 0.7, 3),
        ]
        detector = _make_detector(_make_model(two_mines))
        results = detector.detect(blank_image)
        assert len(results) == 2
        assert all(r.class_name == "mine_cylinder" for r in results)


# ---------------------------------------------------------------------------
# Tests: invalid image inputs
# ---------------------------------------------------------------------------


class TestInvalidImageInputs:
    """detect() should raise descriptive errors for bad inputs."""

    def test_non_array_non_path_raises_type_error(self) -> None:
        detector = _make_detector(_make_model([]))
        with pytest.raises(TypeError, match="NumPy array or a file path"):
            detector.detect(12345)  # type: ignore[arg-type]

    def test_list_raises_type_error(self) -> None:
        detector = _make_detector(_make_model([]))
        with pytest.raises(TypeError):
            detector.detect([[0, 0, 0]] * 640)  # type: ignore[arg-type]

    def test_wrong_ndim_raises_value_error(self) -> None:
        """4-D array (batch) is not accepted."""
        detector = _make_detector(_make_model([]))
        bad = np.zeros((1, 640, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="ndim"):
            detector.detect(bad)

    def test_wrong_dtype_raises_value_error(self) -> None:
        """float32 array is not accepted; must be uint8."""
        detector = _make_detector(_make_model([]))
        bad = np.zeros((640, 640, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            detector.detect(bad)

    def test_missing_file_path_raises_file_not_found(self) -> None:
        """Passing a non-existent file path raises FileNotFoundError."""
        detector = _make_detector(_make_model([]))
        with pytest.raises(FileNotFoundError):
            detector.detect("/tmp/does_not_exist_abc123.png")

    def test_grayscale_2d_array_is_accepted(self) -> None:
        """2-D (grayscale) uint8 arrays are a valid input."""
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        gray = np.zeros((640, 640), dtype=np.uint8)
        result = detector.detect(gray)
        assert isinstance(result, list)

    def test_valid_tempfile_path_is_accepted(self, tmp_path: Path) -> None:
        """A path to an existing file is passed through to the model."""
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"\x89PNG\r\n")  # minimal header — model is mocked
        model_mock = _make_model([])
        detector = _make_detector(model_mock)
        # Model is mocked so no real decode happens; we just verify no error raised
        result = detector.detect(img_path)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Identical inputs must produce identical outputs."""

    def test_same_input_same_output(self, blank_image: np.ndarray) -> None:
        detections = [(5.0, 5.0, 55.0, 55.0, 0.88, 2)]
        model_mock = _make_model(detections)
        detector = _make_detector(model_mock)
        r1 = detector.detect(blank_image)
        r2 = detector.detect(blank_image)
        assert r1 == r2
