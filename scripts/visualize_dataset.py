from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

SPLIT = "train"
IMAGE_DIR = ROOT / SPLIT / "images"
LABEL_DIR = ROOT / SPLIT / "labels"

CLASS_NAMES = {
    0: "submarine_pipeline",
    1: "shipwreck",
    2: "ghost_net",
    3: "mine_cylinder",
}

NUM_IMAGES = 12
OUTPUT = ROOT / "dataset_visualization.jpg"

image_files = list(IMAGE_DIR.glob("*"))

if not image_files:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

random.seed(42)
selected = random.sample(image_files, min(NUM_IMAGES, len(image_files)))

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for ax, image_path in zip(axes, selected):
    image = cv2.imread(str(image_path))

    if image is None:
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            label = CLASS_NAMES.get(class_id, f"class_{class_id}")

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )

    ax.imshow(image, cmap="gray")
    ax.set_title(image_path.name)
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
plt.close()

print(f"Visualization saved to: {OUTPUT}")