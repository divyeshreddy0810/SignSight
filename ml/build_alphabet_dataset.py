"""
Extracts MediaPipe hand landmarks from the ASL alphabet image set into a
compact landmark dataset for the fingerspelling classifier. Only single-letter
class folders are used (del/nothing/space are skipped), subsampled
deterministically to MAX_PER_CLASS images per letter.

Raw landmarks (not features) are stored so the feature definition in
ml/features.py stays the single source of truth — ml/train_letters.py derives
features at training time via letter_features().

The source frame index is stored too, and it matters: the image sets are
sequential video frames of one hand (A1.jpg ... A3000.jpg), so neighbouring
frames are near-duplicates. Random CV over them leaks and reports a fantasy
number; ml/train_letters.py groups by contiguous frame blocks instead.

Output: data/processed_alphabet/{landmarks.npy (N,21,3), is_left.npy (N,),
labels.npy (N,), frame_idx.npy (N,)} plus a per-class report on stdout.

Usage: python ml/build_alphabet_dataset.py
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# First existing layout wins: grassknoted/asl-alphabet nests twice.
RAW_CANDIDATES = [
    ROOT / "data" / "alphabet_raw" / "asl_alphabet_train" / "asl_alphabet_train",
    ROOT / "data" / "alphabet_raw" / "Data",
]
OUT_DIR = ROOT / "data" / "processed_alphabet"
MAX_PER_CLASS = 300  # plenty for a 60-dim static classifier; keeps MediaPipe time sane

mp_hands = mp.solutions.hands


def frame_number(img_path: Path) -> int:
    """A1234.jpg -> 1234. Sorting by this keeps frames in capture order, which
    the blocked CV split in ml/train_letters.py depends on."""
    digits = "".join(c for c in img_path.stem if c.isdigit())
    return int(digits) if digits else 0


def main():
    raw_dir = next((p for p in RAW_CANDIDATES if p.is_dir()), None)
    assert raw_dir is not None, f"no alphabet image set found under {ROOT / 'data' / 'alphabet_raw'}"
    print(f"reading {raw_dir}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    landmarks, is_left, labels, frame_idx = [], [], [], []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        for class_dir in sorted(raw_dir.iterdir()):
            if not class_dir.is_dir() or len(class_dir.name) != 1:
                continue
            letter = class_dir.name
            n_ok = n_fail = 0
            # Evenly spaced across the whole sequence, not the first N frames:
            # a contiguous head would sample one moment of one recording.
            frames = sorted(class_dir.glob("*.jpg"), key=frame_number)
            step = max(1, len(frames) // MAX_PER_CLASS)
            for img_path in frames[::step][:MAX_PER_CLASS]:
                img = cv2.imread(str(img_path))
                if img is None:
                    n_fail += 1
                    continue
                res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not res.multi_hand_landmarks:
                    n_fail += 1
                    continue
                lm = res.multi_hand_landmarks[0].landmark
                landmarks.append([[p.x, p.y, p.z] for p in lm])
                is_left.append(res.multi_handedness[0].classification[0].label == "Left")
                labels.append(letter)
                frame_idx.append(frame_number(img_path))
                n_ok += 1
            print(f"{letter}: {n_ok} ok, {n_fail} no-hand/unreadable", flush=True)

    np.save(OUT_DIR / "landmarks.npy", np.array(landmarks, dtype=np.float32))
    np.save(OUT_DIR / "is_left.npy", np.array(is_left))
    np.save(OUT_DIR / "labels.npy", np.array(labels))
    np.save(OUT_DIR / "frame_idx.npy", np.array(frame_idx))
    print(f"\nDone: {len(labels)} samples, {len(set(labels))} classes -> {OUT_DIR}")


if __name__ == "__main__":
    main()
