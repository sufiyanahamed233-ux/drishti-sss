"""
create_accuracy_test_100.py
---------------------------
Creates a web-uploadable 100-scan accuracy test batch for the Drishti SSS system.

Usage (from project root, venv activated):
    python scripts/create_accuracy_test_100.py

Composition
-----------
    20 submarine_pipeline images
    20 shipwreck images
    20 ghost_net images
    20 mine_cylinder images
    20 true zero-target images (ground-truth label is empty or missing)

What it produces
----------------
    accuracy_test_100/
    |__ scans/           100 unique sonar images (original filenames)
    |__ navigation.csv   frozen-MVP format; rows built from ACTUAL YOLO detections
    |__ ground_truth_manifest.csv   class_id | class_name | gt_count per image

navigation.csv logic
--------------------
- Run best.pt on every selected image at conf=0.25.
- For each predicted detection: one row with simulated lat/lon/heading/altitude
  PLUS slant range and relative bearing derived from the detection bounding box
  (same geometry formulae used by the backend georeferencing engine).
- For images with zero predictions: one scan row with blank detection columns.
- All navigation metadata is SIMULATED (data_source=SIMULATED).

Nothing in the application, backend, frontend, or model is modified.
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
WEIGHTS     = ROOT / "runs" / "detect" / "runs" / "drishti_baseline-3" / "weights" / "best.pt"
TEST_IMAGES = ROOT / "test" / "images"
TEST_LABELS = ROOT / "test" / "labels"
OUT_DIR     = ROOT / "accuracy_test_100"
SCANS_DIR   = OUT_DIR / "scans"

CLASS_NAMES: dict[int, str] = {
    0: "submarine_pipeline",
    1: "shipwreck",
    2: "ghost_net",
    3: "mine_cylinder",
}

PER_CLASS_TARGET = 20
ZERO_TARGET      = 20
CONF_THRESHOLD   = 0.25
RANDOM_SEED      = 7

# Simulated survey (Bay of Bengal)
SIM_LAT     = 13.3400
SIM_LON     = 77.1000
SIM_HEADING = 90.0
SIM_ALT     = 10.0
LAT_STEP    = 0.0001
LON_STEP    = 0.0010

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classes_in_label(label_path: Path) -> set[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return set()
    ids: set[int] = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                ids.add(int(parts[0]))
            except ValueError:
                pass
    return ids


def _gt_count_in_label(label_path: Path) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return counts
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                counts[int(parts[0])] += 1
            except ValueError:
                pass
    return counts


def _select_images() -> list[Path]:
    rng = random.Random(RANDOM_SEED)

    all_images = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
    if not all_images:
        print("[ERROR] No images found in", TEST_IMAGES)
        sys.exit(1)

    # Classify every image
    zero_imgs: list[Path] = []
    class_imgs: dict[int, list[Path]] = defaultdict(list)

    for img in all_images:
        lbl = TEST_LABELS / (img.stem + ".txt")
        classes = _classes_in_label(lbl)
        if not classes:
            zero_imgs.append(img)
        else:
            for cid in classes:
                class_imgs[cid].append(img)

    print(f"  Available zero-target images : {len(zero_imgs)}")
    for cid, name in CLASS_NAMES.items():
        print(f"  Available {name:<25}: {len(class_imgs[cid])}")

    selected: dict[str, Path] = {}

    # Pick per-class images (no overlap between classes)
    for cid in range(len(CLASS_NAMES)):
        candidates = [c for c in class_imgs[cid] if c.stem not in selected]
        rng.shuffle(candidates)
        if len(candidates) < PER_CLASS_TARGET:
            print(f"  [WARN] Only {len(candidates)} images for class {CLASS_NAMES[cid]}")
        for img in candidates[:PER_CLASS_TARGET]:
            selected[img.stem] = img

    # Pick zero-target images (no overlap with class images)
    zero_candidates = [z for z in zero_imgs if z.stem not in selected]
    rng.shuffle(zero_candidates)
    if len(zero_candidates) < ZERO_TARGET:
        print(f"  [WARN] Only {len(zero_candidates)} zero-target images available")
    for img in zero_candidates[:ZERO_TARGET]:
        selected[img.stem] = img

    result = list(selected.values())
    rng.shuffle(result)
    return result


def _bbox_to_geometry(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
    altitude: float,
) -> tuple[float, float]:
    """
    Approximate slant-range and relative-bearing for a predicted bounding box.

    Strategy (matches the backend georeferencing engine conventions):
    - Horizontal centre of bbox gives cross-track position as fraction of image width.
    - Image centre = nadir (0 relative bearing).
    - Left half  = negative relative bearing (port side).
    - Right half = positive relative bearing (starboard).
    - Slant range computed from cross-track distance assuming flat seabed at
      the given altitude.
    - Max half-swath assumed to be 50 m (reasonable for SSS towfish at 10 m alt).
    """
    MAX_HALF_SWATH_M = 50.0

    cx_frac = ((x1 + x2) / 2.0) / img_w   # 0..1
    cross_track_frac = cx_frac - 0.5       # -0.5..+0.5  (negative = port)
    cross_track_m    = cross_track_frac * MAX_HALF_SWATH_M * 2.0

    slant_range_m = math.sqrt(altitude ** 2 + cross_track_m ** 2)
    slant_range_m = max(5.0, min(slant_range_m, 120.0))

    # relative bearing in degrees: 0 = ahead, +90 = starboard, -90 = port
    rel_bearing = math.degrees(math.atan2(cross_track_m, altitude))

    return round(slant_range_m, 1), round(rel_bearing, 1)


def _sep(char: str = "=", width: int = 70) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _sep()
    print("  DRISHTI SSS  -  ACCURACY TEST BATCH GENERATOR  (100 scans)")
    _sep()

    if not WEIGHTS.exists():
        print(f"[ERROR] Trained weights not found:\n  {WEIGHTS}")
        sys.exit(1)
    if not TEST_IMAGES.exists():
        print(f"[ERROR] test/images not found:\n  {TEST_IMAGES}")
        sys.exit(1)

    # 1. Select images
    print("\n[1/5] Selecting 100 images...")
    images = _select_images()
    print(f"  OK {len(images)} images selected")

    # 2. Copy to scans/
    print("\n[2/5] Copying images to accuracy_test_100/scans/...")
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, SCANS_DIR / img.name)
    print(f"  OK {len(images)} images copied")

    # 3. Run YOLO on all 100 images
    print("\n[3/5] Running YOLO11s on all 100 images (conf=0.25)...")
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image as PILImage

    model = YOLO(str(WEIGHTS))

    # image stem -> list of detection dicts
    predictions: dict[str, list[dict]] = {}
    timings: list[float] = []
    total_preds = 0
    pred_class_counts: dict[int, int] = defaultdict(int)

    for i, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        # Get image dimensions for geometry
        try:
            with PILImage.open(img_path) as pil_img:
                img_w, img_h = pil_img.size
        except Exception:
            img_w, img_h = 1024, 1024  # fallback

        dets: list[dict] = []
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy    = boxes.xyxy.cpu().numpy()
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for j in range(len(confs)):
                cid = int(cls_ids[j])
                x1, y1, x2, y2 = float(xyxy[j, 0]), float(xyxy[j, 1]), float(xyxy[j, 2]), float(xyxy[j, 3])
                dets.append({
                    "class_id":   cid,
                    "class_name": CLASS_NAMES.get(cid, f"class_{cid}"),
                    "confidence": round(float(confs[j]), 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_w": img_w, "img_h": img_h,
                })
                pred_class_counts[cid] += 1
                total_preds += 1

        predictions[img_path.stem] = dets

        if i % 10 == 0 or i == len(images):
            avg_ms = sum(timings) / len(timings) * 1000
            print(f"  [{i:3d}/100]  last={elapsed*1000:.1f}ms  avg={avg_ms:.1f}ms")

    total_time = sum(timings)
    print(f"  OK Inference done — {total_preds} detections in {total_time:.2f}s")

    # 4. Build navigation.csv from ACTUAL YOLO detections
    print("\n[4/5] Building navigation.csv from actual detections...")
    rng = random.Random(RANDOM_SEED + 99)

    nav_rows: list[dict] = []
    fieldnames = ["scan_file", "latitude", "longitude", "heading", "altitude",
                  "timestamp", "data_source", "detection_index",
                  "target_range_m", "range_type", "relative_bearing"]

    for idx, img_path in enumerate(images):
        lat      = SIM_LAT + idx * LAT_STEP
        lon      = SIM_LON + idx * LON_STEP
        heading  = SIM_HEADING + rng.uniform(-5.0, 5.0)
        altitude = SIM_ALT    + rng.uniform(-2.0, 2.0)
        minutes  = idx
        timestamp = f"2026-10-04T{8 + minutes // 60:02d}:{minutes % 60:02d}:00Z"

        dets = predictions.get(img_path.stem, [])
        base = dict(
            scan_file=img_path.name,
            latitude=f"{lat:.6f}",
            longitude=f"{lon:.6f}",
            heading=f"{heading:.2f}",
            altitude=f"{altitude:.2f}",
            timestamp=timestamp,
            data_source="SIMULATED",
        )

        if not dets:
            nav_rows.append({**base,
                "detection_index": "", "target_range_m": "",
                "range_type": "", "relative_bearing": ""})
        else:
            for det_idx, det in enumerate(dets):
                slant_m, rel_bear = _bbox_to_geometry(
                    det["x1"], det["y1"], det["x2"], det["y2"],
                    det["img_w"], det["img_h"], altitude,
                )
                nav_rows.append({**base,
                    "detection_index":  str(det_idx),
                    "target_range_m":   str(slant_m),
                    "range_type":       "SLANT",
                    "relative_bearing": str(rel_bear),
                })

    nav_path = OUT_DIR / "navigation.csv"
    with nav_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nav_rows)
    print(f"  OK navigation.csv written ({len(nav_rows)} rows, {len(images)} scans)")

    # 5. Ground-truth manifest
    print("\n[5/5] Building ground_truth_manifest.csv...")
    manifest_rows: list[dict] = []
    for img in images:
        lbl = TEST_LABELS / (img.stem + ".txt")
        gt_counts = _gt_count_in_label(lbl)
        total_gt  = sum(gt_counts.values())
        classes_present = [CLASS_NAMES.get(cid, f"class_{cid}")
                           for cid in sorted(gt_counts.keys())]
        is_zero = (total_gt == 0)
        pred_count = len(predictions.get(img.stem, []))

        manifest_rows.append({
            "scan_file":        img.name,
            "is_zero_target":   str(is_zero).upper(),
            "gt_total_objects": total_gt,
            "gt_classes":       "|".join(classes_present) if classes_present else "none",
            "gt_submarine_pipeline": gt_counts.get(0, 0),
            "gt_shipwreck":          gt_counts.get(1, 0),
            "gt_ghost_net":          gt_counts.get(2, 0),
            "gt_mine_cylinder":      gt_counts.get(3, 0),
            "yolo_predictions":      pred_count,
        })

    manifest_path = OUT_DIR / "ground_truth_manifest.csv"
    manifest_fields = ["scan_file", "is_zero_target", "gt_total_objects",
                       "gt_classes", "gt_submarine_pipeline", "gt_shipwreck",
                       "gt_ghost_net", "gt_mine_cylinder", "yolo_predictions"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  OK ground_truth_manifest.csv written ({len(manifest_rows)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    zero_count  = sum(1 for r in manifest_rows if r["is_zero_target"] == "TRUE")
    class_img_counts: dict[int, int] = defaultdict(int)
    for img in images:
        lbl = TEST_LABELS / (img.stem + ".txt")
        for cid in _classes_in_label(lbl):
            class_img_counts[cid] += 1

    print()
    _sep()
    print("  BATCH SUMMARY  -  accuracy_test_100/")
    _sep()
    print(f"\n  Composition")
    for cid, name in CLASS_NAMES.items():
        print(f"    {name:<25} : {class_img_counts.get(cid, 0):2d} images")
    print(f"    {'zero-target':<25} : {zero_count:2d} images")
    print(f"    {'TOTAL':<25} : {len(images):2d} images")

    avg_ms = total_time / len(timings) * 1000
    print(f"\n  Inference (drishti_baseline-3/best.pt, conf={CONF_THRESHOLD})")
    print(f"    Total time             : {total_time:.2f} s")
    print(f"    Average / image        : {avg_ms:.1f} ms")
    print(f"    Total detections       : {total_preds}")
    for cid, name in CLASS_NAMES.items():
        print(f"      {name:<25} : {pred_class_counts.get(cid, 0)}")

    print(f"\n  Output files")
    print(f"    accuracy_test_100/scans/                ({len(images)} images)")
    print(f"    accuracy_test_100/navigation.csv        ({len(nav_rows)} rows)")
    print(f"    accuracy_test_100/ground_truth_manifest.csv")

    print(f"\n  Upload instructions")
    print(f"    1. Go to http://localhost:5173/batch-upload")
    print(f"    2. Upload all 100 files from accuracy_test_100/scans/")
    print(f"    3. Upload accuracy_test_100/navigation.csv")
    print(f"    4. Click Analyze Investigation")
    print(f"    5. Compare web results against ground_truth_manifest.csv")

    _sep()
    print("  Done.")
    _sep()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()