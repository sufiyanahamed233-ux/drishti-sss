---
license: cc-by-sa-4.0
task_categories:
  - object-detection
tags:
  - sonar
  - side-scan-sonar
  - marine-debris
  - underwater
  - ghost-net
  - shipwreck
size_categories:
  - 1K<n<10K
---

# DRISHTI — side-scan sonar training splits

The assembled, preprocessed train / val / test tiles behind the
[DRISHTI detector](https://github.com/Rehan9599/Sonar-Drishti) — an SIH 2026 (PS 26057)
marine-debris and anomaly detector for side-scan sonar. YOLO format, 640 px tiles.

## Licence — CC-BY-SA-4.0

This release is **Creative Commons Attribution-ShareAlike 4.0 International**. You may
share and adapt it, including commercially, provided you (a) **credit** the upstream
sources below and this repo, and (b) release any derivative dataset under
**CC-BY-SA-4.0** as well.

The ShareAlike term is inherited: two of the source datasets (`mine`, and the
`crab_pot` set that is *excluded* here) are themselves CC-BY-SA-4.0, so the assembled
whole must be too.

### Attribution (required)

| Portion | Source | Licence |
|---|---|---|
| `pipe`, `bg` | [SubPipe / SubPipeMini2](https://zenodo.org/records/10808161) — Álvarez-Tuñón et al., OceanScan-MST | CC-BY-4.0 |
| `wreckA` | [AI4Shipwrecks](https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x) — Sethuraman et al., UM Field Robotics / NOAA Thunder Bay | CC-BY-4.0 |
| `wreckR` | [Side Scan Sonar (Ship, Plane)](https://universe.roboflow.com/dae-hyeok-lee/side-scan-sonar) — Dae Hyeok Lee, Roboflow Universe | CC-BY-4.0 |
| `mine` | [Sonar Imaging Mine Detection](https://www.kaggle.com/datasets/sierra022/sonar-imaging-mine-detection) — MILCO contacts | CC-BY-SA-4.0 |
| `synth` | procedural acoustic generator (`ml/scripts/build_synthetic_data.py`) | original work, CC-BY-SA-4.0 with the rest |

The `synth` background canvases are drawn from the Roboflow SSS set (CC-BY-4.0), then
composited with procedurally modelled objects and filtered — the source imagery is not
recognisable in the output.

## What is *not* in this release

- **`crab_pot`** (class id `0`) — 1,200 tiles were used to train the shipped model as a
  hard negative, from a HuggingFace set that is now access-gated. They are **omitted
  here**. Class `0` therefore has zero examples in this repo. To reproduce the exact
  training set, rebuild from the GitHub `ml/scripts/` recipe.
- **KLSG / SeabedObjects (Ship & Airplane)** — never contributed a tile (`KLSG_TRAIN_CAP = 0`
  in `build_dataset.py`); its weak full-frame boxes hurt shipwreck precision. Listed only
  so the exclusion is on the record.

## Contents

| Split | Images | Labels |
|---|---|---|
| train | 3,875 | 3,875 |
| val   | 630   | 630   |
| test  | 700   | 700   |
| **total** | **5,205** | **5,205** |

~1.8 GB. Layout:

```
train/images/*.jpg   train/labels/*.txt
val/images/*.jpg     val/labels/*.txt
test/images/*.jpg    test/labels/*.txt
drishti.yaml         # Ultralytics data config
```

Labels are YOLO boxes: `class_id x_center y_center width height` (normalised).

## Classes

| id | class | in this release |
|---|---|---|
| 0 | `crab_pot` | — (excluded, see above) |
| 1 | `submarine_pipeline` | ✓ |
| 2 | `shipwreck` | ✓ |
| 3 | `ghost_net` | ✓ (100 % synthetic) |
| 4 | `mine_cylinder` | ✓ |

The shipped product uses classes 1–4. `ghost_net` is fully synthetic — no public real
ghost-net-in-SSS dataset exists; a Microsoft AI for Good / WWF effort had 412 real
segments total and called it a feasibility study.

## Preprocessing — already applied

Every tile has been through **Lee speckle filter + CLAHE** (`despeckle_clahe()` in
`ml/scripts/preprocess_sonar.py`, clip 3.0, 8×8 grid, 7×7 Lee kernel). Do **not** apply
it again. If you train on these tiles, apply the same filter to your inference inputs.

Speckle in sonar is *multiplicative* (`I_obs = I_true · n`), so a plain blur destroys the
object and shadow edges that carry the signal. The Lee filter is a local MMSE estimate
that smooths flat seabed and preserves edges; CLAHE lifts faint contrast with a clip
limit so flat-sand speckle is not amplified.

Note: an ablation found this preprocessing gave **no accuracy gain** over raw tiles once
training-time augmentation was strong — it is retained because CLAHE'd input makes
acoustic shadows more detectable for the downstream geometry check.

## Provenance — per prefix (train split)

Filenames carry their source.

| Prefix | Count | Source | Nature |
|---|---|---|---|
| `synth` | 1,250 | procedural generator, modelled acoustic physics | original work |
| `pipe`  | 1,000 | SubPipeMini2 survey strips, tiled | real |
| `wreckA`| 546   | AI4Shipwrecks transects (pixel masks → boxes) | real |
| `bg`    | 500   | object-free SubPipeMini2 tiles (hard negatives) | real |
| `wreckR`| 354   | Roboflow side-scan-sonar (Ship + Plane) | real |
| `mine`  | 225   | Kaggle sonar-mine (MILCO contacts) | real |

Full source detail: `docs/PROJECT_RECORD.html` §03 and §15 in the GitHub repo.

## Known caveats

- **Shipwreck split not yet audited site-disjoint.** An audit to guarantee no wreck
  *site* appears in both train and test is pending; treat shipwreck metrics as optimistic
  until it lands.
- **The test set is deliberately hard.** A 50 %-overlap re-tile tripled shipwreck test
  instances with partial and near-duplicate tiles. Numbers on this split are not
  comparable to papers using a non-overlapping tiling of the same source.
- **`ghost_net` evaluation is synthetic-on-synthetic.** Not a field number.
- **Class imbalance is deliberate.** Per-class caps per split; a dominant class suppresses
  accuracy on feature-dissimilar classes.

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download("rehan9599/drishti-sss", repo_type="dataset",
                  local_dir="ml/data/splits")
```

```bash
yolo detect train data=drishti.yaml model=yolov8s.pt imgsz=640 epochs=120 batch=16
```

## Citation

```bibtex
@software{drishti2026,
  title  = {DRISHTI: AI-Powered Marine Debris Detection from Side-Scan Sonar},
  author = {Fazal, Rehan and others},
  year   = {2026},
  note   = {Smart India Hackathon 2026, Problem Statement 26057},
  url    = {https://github.com/Rehan9599/Sonar-Drishti}
}
```

Please also cite the upstream datasets listed under **Attribution**.
