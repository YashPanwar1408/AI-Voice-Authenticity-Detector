"""
data_loader.py
--------------
Loads an audio dataset from the following folder structure:

    data/audio/
        REAL/   -> label 0
        FAKE/   -> label 1

Usage:
    from src.data_loader import load_data
    file_paths, labels = load_data()
"""

import os


def load_data(data_dir: str = "data/audio") -> tuple[list[str], list[int]]:
    """
    Iterate through REAL/ and FAKE/ sub-folders inside `data_dir`,
    collect all .wav file paths, and assign integer labels.

    Args:
        data_dir: Root directory that contains REAL/ and FAKE/ sub-folders.
                  Defaults to "data/audio".

    Returns:
        file_paths: List of paths (str) to every .wav file found.
        labels:     Matching list of integer labels (0 = REAL, 1 = FAKE).
    """
    class_map = {
        "REAL": 0,
        "FAKE": 1,
    }

    file_paths: list[str] = []
    labels: list[int] = []

    for class_name, label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARNING] Folder not found, skipping: {class_dir}")
            continue

        for filename in sorted(os.listdir(class_dir)):
            if filename.lower().endswith(".wav"):
                full_path = os.path.join(class_dir, filename)
                file_paths.append(full_path)
                labels.append(label)

    # ── Stats ──────────────────────────────────────────────────────────────
    real_count = labels.count(0)
    fake_count = labels.count(1)
    total = len(labels)

    print(f"{'─' * 38}")
    print(f"  Total samples   : {total}")
    print(f"  Real  (label 0) : {real_count}")
    print(f"  Fake  (label 1) : {fake_count}")
    print(f"{'─' * 38}")

    return file_paths, labels
