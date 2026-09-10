"""
Hand-crafted feature extraction for the classical ML baseline.

Each clip is a (WINDOW_SIZE, 75, 4) landmark sequence: 33 pose + 21 left-hand +
21 right-hand keypoints, each [x, y, z, visibility]. Summary statistics
(mean/std/range/velocity) are computed per keypoint per coordinate, which is
the standard hand-crafted representation used in classical gesture-recognition
baselines (vs. the raw-sequence input the LSTM uses). Restricted to hand
keypoints + arm-relevant pose keypoints (wrists/elbows/shoulders) rather than
all 33 pose landmarks, since the dataset has ~15-20 samples/class and a
1000+-dim feature space would overfit before any model-specific regularization
gets a chance to help.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = PROCESSED_DIR / "labels.csv"

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

# MediaPipe Pose landmark indices for shoulders/elbows/wrists (arms only).
ARM_POSE_INDICES = [11, 12, 13, 14, 15, 16]  # L/R shoulder, L/R elbow, L/R wrist

HAND_START = NUM_POSE_LANDMARKS
LEFT_HAND_INDICES = list(range(HAND_START, HAND_START + NUM_HAND_LANDMARKS))
RIGHT_HAND_INDICES = list(range(HAND_START + NUM_HAND_LANDMARKS, HAND_START + 2 * NUM_HAND_LANDMARKS))

# Hands only: the landmark-subset ablation (ml/evaluate.py, landmark_subset
# sweep in results/evaluation_results.csv) shows the discriminative signal for
# this vocabulary is carried by the hands; arm keypoints add dimensions
# without adding information.
FEATURE_KEYPOINT_INDICES = LEFT_HAND_INDICES + RIGHT_HAND_INDICES

# MediaPipe Pose left/right landmark pairs (index 0 = nose is its own mirror).
POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                 (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                 (27, 28), (29, 30), (31, 32)]


def clip_features(sequence: np.ndarray) -> np.ndarray:
    """v1 (absolute coordinates): (T, 75, 4) -> 1D feature vector."""
    kp = sequence[:, FEATURE_KEYPOINT_INDICES, :3]  # (T, K, 3) — drop visibility for stats
    mean = kp.mean(axis=0)                      # (K, 3)
    std = kp.std(axis=0)                         # (K, 3)
    rng = kp.max(axis=0) - kp.min(axis=0)        # (K, 3)
    velocity = np.abs(np.diff(kp, axis=0)).mean(axis=0)  # (K, 3) — mean frame-to-frame motion
    return np.concatenate([mean.ravel(), std.ravel(), rng.ravel(), velocity.ravel()])


def mirror(sequences: np.ndarray) -> np.ndarray:
    """Horizontal flip of a (N, T, 75, 4) batch: reflect x, swap L/R pose pairs
    and hand blocks. Absent keypoints (visibility == 0) keep x = 0 rather than
    1 - 0."""
    out = sequences.copy()
    detected = out[..., 3] > 0
    out[..., 0] = np.where(detected, 1.0 - out[..., 0], out[..., 0])
    for left_idx, right_idx in POSE_LR_PAIRS:
        out[:, :, [left_idx, right_idx]] = out[:, :, [right_idx, left_idx]]
    out[:, :, LEFT_HAND_INDICES + RIGHT_HAND_INDICES] = out[:, :, RIGHT_HAND_INDICES + LEFT_HAND_INDICES]
    return out


def canonicalize(sequences: np.ndarray) -> np.ndarray:
    """Mirror clips whose LEFT hand is dominant (detected in more frames) so
    every clip is right-hand-dominant. Deterministic and per-clip, applied
    identically at training and serving time — handedness invariance without
    augmentation. Idempotent."""
    out = sequences.copy()
    left = (sequences[:, :, LEFT_HAND_INDICES, 3] > 0).any(axis=2).sum(axis=1)
    right = (sequences[:, :, RIGHT_HAND_INDICES, 3] > 0).any(axis=2).sum(axis=1)
    flip = left > right
    if flip.any():
        out[flip] = mirror(sequences[flip])
    return out


def features_v2(sequences: np.ndarray) -> np.ndarray:
    """v2 (normalized): (N, T, 75, 4) -> (N, 504) feature matrix.

    Per frame and hand: wrist-relative keypoint offsets (hand shape, decoupled
    from where the signer stands) plus the wrist position relative to the nose,
    scaled by shoulder width (sign location in body space). Frames without a
    detected hand contribute exact zeros. Same four summary statistics as v1.
    """
    n, t = sequences.shape[:2]
    nose = sequences[:, :, 0, :3]
    shoulder_dist = np.linalg.norm(sequences[:, :, 11, :3] - sequences[:, :, 12, :3], axis=-1)
    scale = np.where(shoulder_dist.mean(axis=1) > 1e-3, shoulder_dist.mean(axis=1), 1.0)[:, None, None]
    parts = []
    for hand in (LEFT_HAND_INDICES, RIGHT_HAND_INDICES):
        block = sequences[:, :, hand, :]
        present = (block[..., 3] > 0).any(axis=2)[..., None]      # (N, T, 1)
        wrist = block[:, :, 0, :3]
        shape = (block[:, :, 1:, :3] - wrist[:, :, None, :]) * present[..., None]
        location = (wrist - nose) / scale * present
        parts.append(shape.reshape(n, t, -1))
        parts.append(location)
    per_frame = np.concatenate(parts, axis=2)                     # (N, T, 126)
    return np.concatenate([per_frame.mean(axis=1), per_frame.std(axis=1),
                           per_frame.max(axis=1) - per_frame.min(axis=1),
                           np.abs(np.diff(per_frame, axis=1)).mean(axis=1)], axis=1)


def clip_features_v2(sequence: np.ndarray) -> np.ndarray:
    """Serving-side twin of features_v2 for a single (T, 75, 4) clip."""
    return features_v2(sequence[None])[0]


MIDDLE_MCP = 9  # MediaPipe hand landmark: base of the middle finger
INDEX_MCP, PINKY_MCP = 5, 17
# MediaPipe hand landmark chains, wrist-outwards, one per finger.
FINGER_CHAINS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]


def letter_features(hand_xyz: np.ndarray, is_left: bool, aspect: float = 1.0) -> np.ndarray:
    """Static fingerspelling descriptor for one (21, 3) hand. 40 dims.

    Wrist-centred keypoint offsets, x multiplied by the image aspect ratio so
    coordinates are isotropic regardless of frame shape (training crops are
    square, the webcam is 4:3), mirrored to right-hand form when is_left, and
    scale-normalised by the wrist-to-middle-MCP palm length (invariant to how
    close the hand is to the camera).

    x and y only: z is estimated differently by the standalone Hands model
    (used to build the training set) and by Holistic (used at serve time), so
    including it imports a train/serve mismatch for no gain — dropping z scored
    slightly HIGHER in blocked CV (0.946 vs 0.941).

    is_left still normalises handedness, but the deployed model is trained on
    both handedness forms (ml/train_letters.py), so a wrong is_left no longer
    wrecks the prediction — Holistic mislabels it often when one hand is
    visible, and that cliff was 0.94 -> 0.56.
    """
    off = (hand_xyz[1:, :2] - hand_xyz[0, :2]).astype(np.float32)
    off[:, 0] *= aspect
    if is_left:
        off[:, 0] = -off[:, 0]
    palm = hand_xyz[MIDDLE_MCP] - hand_xyz[0]
    scale = float(np.linalg.norm([palm[0] * aspect, palm[1], palm[2]]))
    if scale < 1e-6:
        scale = 1.0
    return (off / scale).ravel()


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros_like(v)


def letter_features_v2(hand_xyz: np.ndarray, is_left: bool, aspect: float = 1.0) -> np.ndarray:
    """Rotation-invariant fingerspelling descriptor for one (21, 3) hand. 75 dims.

    letter_features() measures offsets in screen axes, so the same handshape
    tilted reads as different numbers — training images hold one orientation,
    live signers do not. Here the hand is expressed in its OWN frame instead:
    the palm basis (wrist->middle-MCP as the up axis, palm normal from the
    index/pinky MCP spread) is built per sample, keypoint offsets are rotated
    into it, and the result is scaled by palm length. Whole-hand rotation
    therefore cancels out; only the handshape survives.

    Appends 3 joint angles per finger (15 total). These are redundant with the
    coordinates in principle, but they state finger curl — the thing that
    separates letters — directly rather than implicitly.
    """
    pts = hand_xyz.astype(np.float32).copy()
    pts[:, 0] *= aspect
    if is_left:
        pts[:, 0] = -pts[:, 0]  # to right-hand form, before the frame is built

    off = pts - pts[0]
    y_axis = _unit(off[MIDDLE_MCP])
    normal = _unit(np.cross(off[INDEX_MCP], off[PINKY_MCP]))
    x_axis = _unit(np.cross(y_axis, normal))
    z_axis = np.cross(x_axis, y_axis)
    if not np.any(x_axis) or not np.any(y_axis):
        # Degenerate (collinear or missing) landmarks: fall back to raw axes
        # rather than emitting NaNs into the forest.
        basis = np.eye(3, dtype=np.float32)
    else:
        basis = np.stack([x_axis, y_axis, z_axis])

    scale = float(np.linalg.norm(off[MIDDLE_MCP]))
    if scale < 1e-6:
        scale = 1.0
    local = (off[1:] @ basis.T) / scale

    angles = []
    for chain in FINGER_CHAINS:
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            v1, v2 = _unit(pts[b] - pts[a]), _unit(pts[c] - pts[b])
            angles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    return np.concatenate([local.ravel(), np.array(angles, dtype=np.float32)])


def augment_hand(hand_xyz: np.ndarray, rng, max_deg: float = 18.0,
                 sigma: float = 0.004) -> np.ndarray:
    """Landmark-space augmentation: a small random 3D rotation plus per-keypoint
    jitter. Rotation buys tolerance to hand tilt the single-setup training
    images never show; jitter models MediaPipe's own landmark noise."""
    axis = _unit(rng.normal(size=3).astype(np.float32))
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg))
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    centre = hand_xyz[0]
    out = (hand_xyz - centre) @ R.T + centre
    return (out + rng.normal(0, sigma, out.shape)).astype(np.float32)


def load_dataset():
    """Returns (X, y, groups) — groups is signer_id, for subject-independent CV splits."""
    labels = pd.read_csv(LABELS_CSV)
    X, y, groups = [], [], []
    for _, row in labels.iterrows():
        npy_path = PROCESSED_DIR / f"{row['clip_id']}.npy"
        if not npy_path.exists():
            continue
        sequence = np.load(npy_path)
        X.append(clip_features(sequence))
        y.append(row["gloss"])
        groups.append(row["signer_id"])
    return np.array(X), np.array(y), np.array(groups)


if __name__ == "__main__":
    X, y, groups = load_dataset()
    print(f"X shape: {X.shape}, classes: {sorted(set(y))}, unique signers: {sorted(set(groups))}")
