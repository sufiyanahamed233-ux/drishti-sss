from pathlib import Path
import shutil

DATASET = Path(__file__).resolve().parent.parent

SPLITS = ["train", "val", "test"]

# Original DRISHTI IDs -> new IDs
CLASS_MAP = {
    1: 0,  # submarine_pipeline
    2: 1,  # shipwreck
    3: 2,  # ghost_net
    4: 3,  # mine_cylinder
}

for split in SPLITS:
    labels_dir = DATASET / split / "labels"
    backup_dir = DATASET / split / "labels_backup"

    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    if not backup_dir.exists():
        shutil.copytree(labels_dir, backup_dir)
        print(f"Backup created: {backup_dir}")

    changed = 0
    skipped = 0

    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()
        new_lines = []

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label in {label_file}: {line}"
                )

            old_id = int(parts[0])

            if old_id not in CLASS_MAP:
                raise ValueError(
                    f"Unexpected class ID {old_id} in {label_file}"
                )

            parts[0] = str(CLASS_MAP[old_id])
            new_lines.append(" ".join(parts))

        label_file.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else "")
        )

        changed += 1

    print(f"{split}: remapped {changed} label files")

print("\nLabel remapping completed successfully.")
print("Original labels are backed up in train/val/test/labels_backup/")