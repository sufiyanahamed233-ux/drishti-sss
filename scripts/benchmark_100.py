"""
benchmark_100.py
----------------
Standalone benchmark for the trained Drishti YOLO11s model.

Usage (from project root, venv activated):
    python scripts/benchmark_100.py

Steps
-----
1. Select 100 images from test/images, balanced at 25 per class.
2. Copy to benchmark_100/scans/  (original filenames preserved).
3. Copy matching labels to benchmark_100/ground_truth/
4. Generate benchmark_100/navigation.csv  (simulated, frozen-MVP format).
5. Run drishti_baseline-3/best.pt on all 100 images; report timing.
6. Compute mAP50 / mAP50-95 / P / R via Ultralytics val().
7. Print concise benchmark summary.

No application source files are modified.
"""

from __future__ import annotations

import csv
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
BENCH_DIR   = ROOT / "benchmark_100"
SCANS_DIR   = BENCH_DIR / "scans"
GT_DIR      = BENCH_DIR / "ground_truth"

CLASS_NAMES: dict[int, str] = {
    0: "submarine_pipeline",
    1: "shipwreck",
    2: "ghost_net",
    3: "mine_cylinder",
}

TOTAL_IMAGES     = 100
TARGET_PER_CLASS = TOTAL_IMAGES // len(CLASS_NAMES)   # 25 each
CONF_THRESHOLD   = 0.25
RANDOM_SEED      = 42

# Simulated survey origin (Bay of Bengal)
SIM_LAT_ORIGIN     = 13.3400
SIM_LON_ORIGIN     = 77.1000
SIM_HEADING_BASE   = 90.0
SIM_ALTITUDE_BASE  = 10.0
SIM_ALTITUDE_JITTER= 2.0
SIM_HEADING_JITTER = 5.0
LAT_STEP = 0.0001
LON_STEP = 0.0010

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classes_in_label(label_path: Path) -> set[int]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return set()
    classes: set[int] = set()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                pass
    return classes


def _select_images() -> list[Path]:
    rng = random.Random(RANDOM_SEED)

    all_images = sorted(TEST_IMAGES.glob("*.jpg")) + sorted(TEST_IMAGES.glob("*.png"))
    if not all_images:
        print("[ERROR] No images found in", TEST_IMAGES)
        sys.exit(1)

    stem_classes: dict[str, set[int]] = {}
    for img in all_images:
        lbl = TEST_LABELS / (img.stem + ".txt")
        stem_classes[img.stem] = _classes_in_label(lbl)

    per_class: dict[int, list[Path]] = defaultdict(list)
    for img in all_images:
        for cid in stem_classes[img.stem]:
            per_class[cid].append(img)

    selected_stems: dict[str, Path] = {}
    for cid in range(len(CLASS_NAMES)):
        candidates = [c for c in per_class[cid] if c.stem not in selected_stems]
        rng.shuffle(candidates)
        for img in candidates[:TARGET_PER_CLASS]:
            selected_stems[img.stem] = img

    if len(selected_stems) < TOTAL_IMAGES:
        remaining = [img for img in all_images if img.stem not in selected_stems]
        rng.shuffle(remaining)
        for img in remaining:
            if len(selected_stems) >= TOTAL_IMAGES:
                break
            selected_stems[img.stem] = img

    result = list(selected_stems.values())[:TOTAL_IMAGES]
    rng.shuffle(result)
    return result


def _build_navigation_csv(images: list[Path], out_path: Path) -> None:
    rng = random.Random(RANDOM_SEED + 1)
    rows: list[dict] = []

    for idx, img in enumerate(images):
        lat      = SIM_LAT_ORIGIN + idx * LAT_STEP
        lon      = SIM_LON_ORIGIN + idx * LON_STEP
        heading  = SIM_HEADING_BASE  + rng.uniform(-SIM_HEADING_JITTER,  SIM_HEADING_JITTER)
        altitude = SIM_ALTITUDE_BASE + rng.uniform(-SIM_ALTITUDE_JITTER, SIM_ALTITUDE_JITTER)
        minutes  = idx
        timestamp = f"2026-10-04T{8 + minutes // 60:02d}:{minutes % 60:02d}:00Z"

        lbl = TEST_LABELS / (img.stem + ".txt")
        det_lines = []
        if lbl.exists() and lbl.stat().st_size > 0:
            det_lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]

        base = dict(
            scan_file=img.name,
            latitude=f"{lat:.6f}",
            longitude=f"{lon:.6f}",
            heading=f"{heading:.2f}",
            altitude=f"{altitude:.2f}",
            timestamp=timestamp,
            data_source="SIMULATED",
        )
        if not det_lines:
            rows.append({**base, "detection_index": "", "target_range_m": "", "range_type": "", "relative_bearing": ""})
        else:
            for det_idx in range(len(det_lines)):
                rows.append({**base,
                    "detection_index":  str(det_idx),
                    "target_range_m":   f"{rng.uniform(20.0, 80.0):.1f}",
                    "range_type":       "SLANT",
                    "relative_bearing": f"{rng.uniform(-45.0, 45.0):.1f}",
                })

    fieldnames = ["scan_file", "latitude", "longitude", "heading", "altitude",
                  "timestamp", "data_source", "detection_index", "target_range_m",
                  "range_type", "relative_bearing"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  OK navigation.csv written  ({len(rows)} rows, {len(images)} scans)")


def _ensure_dir_link(src: Path, dst: Path) -> None:
    """Create dst as a symlink to src, or copy if symlinks not allowed."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def _sep(char: str = "=", width: int = 70) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _sep()
    print("  DRISHTI SSS  -  YOLO11s  -  100-IMAGE BENCHMARK")
    _sep()

    # Pre-flight
    if not WEIGHTS.exists():
        print(f"[ERROR] Trained weights not found:\n  {WEIGHTS}")
        sys.exit(1)
    if not TEST_IMAGES.exists():
        print(f"[ERROR] test/images not found:\n  {TEST_IMAGES}")
        sys.exit(1)

    # 1. Select images
    print("\n[1/5] Selecting 100 images (target: 25 per class)...")
    images = _select_images()
    print(f"  OK Selected {len(images)} images")
    class_counts: dict[int, int] = defaultdict(int)
    for img in images:
        for cid in _classes_in_label(TEST_LABELS / (img.stem + ".txt")):
            class_counts[cid] += 1
    for cid, name in CLASS_NAMES.items():
        print(f"     {name:<25} : {class_counts.get(cid, 0)} images containing this class")

    # 2. Copy scans + ground truth
    print("\n[2/5] Copying images and ground-truth labels...")
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    copied_imgs = copied_lbls = 0
    for img in images:
        shutil.copy2(img, SCANS_DIR / img.name)
        copied_imgs += 1
        lbl = TEST_LABELS / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, GT_DIR / lbl.name)
            copied_lbls += 1
    print(f"  OK {copied_imgs} images  -> benchmark_100/scans/")
    print(f"  OK {copied_lbls} labels  -> benchmark_100/ground_truth/")

    # 3. navigation.csv
    print("\n[3/5] Generating navigation.csv...")
    _build_navigation_csv(images, BENCH_DIR / "navigation.csv")

    # 4. Inference timing
    print("\n[4/5] Running YOLO11s inference on 100 images...")
    print(f"  Weights : {WEIGHTS.relative_to(ROOT)}")
    print(f"  Conf    : {CONF_THRESHOLD}")

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    timings: list[float] = []
    total_detections = 0
    pred_counts: dict[int, int]        = defaultdict(int)
    pred_conf:   dict[int, list[float]]= defaultdict(list)

    for i, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            total_detections += len(confs)
            for j in range(len(confs)):
                cid = int(cls_ids[j])
                pred_counts[cid] += 1
                pred_conf[cid].append(float(confs[j]))

        if i % 10 == 0 or i == len(images):
            avg_so_far = sum(timings) / len(timings)
            print(f"  [{i:3d}/100]  last={elapsed*1000:.1f}ms  avg={avg_so_far*1000:.1f}ms")

    # 5. Ultralytics val() metrics
    # Ultralytics resolves labels by replacing '/images/' with '/labels/'
    # in the dataset path.  Create the expected directory structure:
    #   benchmark_100/images/ -> scans/   (symlink or copy)
    #   benchmark_100/labels/ -> ground_truth/  (symlink or copy)
    print("\n[5/5] Computing detection metrics via Ultralytics val()...")
    _ensure_dir_link(SCANS_DIR, BENCH_DIR / "images")
    _ensure_dir_link(GT_DIR,    BENCH_DIR / "labels")

    bench_yaml = BENCH_DIR / "benchmark.yaml"
    bench_yaml.write_text(
        f"path: {BENCH_DIR.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        + "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
        + "\n"
    )

    # Delete any stale val cache so Ultralytics re-scans labels
    for stale in BENCH_DIR.glob("*.cache"):
        stale.unlink(missing_ok=True)

    try:
        val_metrics = model.val(
            data=str(bench_yaml),
            split="val",
            conf=CONF_THRESHOLD,
            iou=0.5,
            verbose=False,
            plots=False,
            save=False,
            save_json=False,
            workers=0,    # Windows: prevents multiprocessing DataLoader hang
            batch=8,
        )
        map50     = val_metrics.box.map50
        map50_95  = val_metrics.box.map
        precision = val_metrics.box.mp
        recall    = val_metrics.box.mr
        metrics_ok = True
    except Exception as exc:
        print(f"  [WARN] val() failed: {exc}")
        map50 = map50_95 = precision = recall = float("nan")
        metrics_ok = False

    # Summary
    total_time = sum(timings)
    avg_time   = total_time / len(timings)
    min_time   = min(timings)
    max_time   = max(timings)

    print()
    _sep()
    print("  BENCHMARK SUMMARY  -  drishti_baseline-3 / best.pt")
    _sep()

    print(f"\n  Dataset")
    print(f"    Images evaluated       : {len(images)}")
    for cid, name in CLASS_NAMES.items():
        print(f"      {name:<25} : {class_counts.get(cid, 0)} images with class")

    print(f"\n  Inference Timing  (pure forward pass)")
    print(f"    Total time             : {total_time*1000:.1f} ms  ({total_time:.2f} s)")
    print(f"    Average / image        : {avg_time*1000:.2f} ms")
    print(f"    Min / Max              : {min_time*1000:.2f} ms / {max_time*1000:.2f} ms")
    print(f"    Throughput             : {len(images)/total_time:.1f} img/s")

    print(f"\n  YOLO Predictions  (conf >= {CONF_THRESHOLD})")
    print(f"    Total detections       : {total_detections}")
    for cid, name in CLASS_NAMES.items():
        count = pred_counts.get(cid, 0)
        confs  = pred_conf.get(cid, [])
        avg_c  = sum(confs) / len(confs) if confs else 0.0
        print(f"      {name:<25} : {count:4d}  (avg conf {avg_c:.3f})")

    print(f"\n  Detection Metrics  (IoU=0.50, vs ground-truth labels)")
    if metrics_ok:
        print(f"    mAP@0.50               : {map50:.4f}")
        print(f"    mAP@0.50:0.95          : {map50_95:.4f}")
        print(f"    Precision (mean)       : {precision:.4f}")
        print(f"    Recall    (mean)       : {recall:.4f}")
    else:
        print("    mAP metrics            : unavailable (val() error above)")

    print(f"\n  Output files")
    print(f"    benchmark_100/scans/          ({copied_imgs} images)")
    print(f"    benchmark_100/ground_truth/   ({copied_lbls} labels)")
    print(f"    benchmark_100/navigation.csv")
    print(f"    benchmark_100/benchmark.yaml")

    _sep()
    print("  Done.")
    _sep()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()