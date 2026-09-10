"""
Smallest checks that fail if the pipeline breaks. No framework needed.

Covers the contracts that connect training to serving: feature dimensionality,
mirror() being a true involution, smoothing preserving shape, and the deployed
model matching the shared feature path (the check that catches a clobbered or
stale ml/models/random_forest.joblib at test time instead of serve time).

Usage: python ml/test_smoke.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import smooth
from features import (augment_hand, canonicalize, clip_features, clip_features_v2,
                      letter_features, letter_features_v2, mirror)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest.joblib"


def main():
    rng = np.random.default_rng(0)
    seq = rng.random((30, 75, 4)).astype(np.float32)
    seq[..., 3] = 1.0
    seq[:, 40:50] = 0.0  # absent keypoints: mirror must leave x = 0, not 1 - 0

    assert clip_features(seq).shape == (504,), "v1 feature dim changed"

    assert np.allclose(mirror(mirror(seq[None]))[0], seq, atol=1e-6), "mirror is not an involution"
    # absent left-hand keypoints (40:50) swap into the right-hand block (61:71)
    # and must stay exactly zero there — x = 0, not 1 - 0.
    assert (mirror(seq[None])[0, :, 61:71] == 0).all(), "mirror corrupted absent keypoints"

    canon = canonicalize(seq[None])
    assert np.array_equal(canonicalize(canon), canon), "canonicalize is not idempotent"

    # v2 features: absent hand contributes exact zeros in its shape+location dims
    absent = seq.copy()
    absent[:, 33:54] = 0.0  # no left hand in any frame
    fv = clip_features_v2(absent)
    assert fv.shape == (504,), "v2 feature dim changed"
    assert (np.abs(fv.reshape(4, 126)[:, :63]) < 1e-7).all(), "absent hand leaked into v2 features"

    assert smooth(seq[None]).shape == (1, 30, 75, 4), "smooth changed shape"

    # deployed model must match the serving feature path exactly
    model = joblib.load(MODEL_PATH)
    served = clip_features_v2(canonicalize(seq[None])[0])
    assert model.n_features_in_ == served.shape[0], (
        f"deployed model expects {model.n_features_in_} features, serving path yields {served.shape[0]}")
    probs = model.predict_proba(served.reshape(1, -1))
    assert probs.shape[1] == 8 and abs(probs.sum() - 1.0) < 1e-6

    # fingerspelling features: shape, left-mirror = negated x offsets, scale invariance
    hand = rng.random((21, 3)).astype(np.float32)
    lf = letter_features(hand, is_left=False)
    lf_left = letter_features(hand, is_left=True)
    assert lf.shape == (40,), "letter feature dim changed"
    assert np.allclose(lf_left.reshape(20, 2)[:, 0], -lf.reshape(20, 2)[:, 0], atol=1e-6)
    assert np.allclose(lf_left.reshape(20, 2)[:, 1], lf.reshape(20, 2)[:, 1], atol=1e-6)
    assert np.allclose(letter_features(hand * 2.0, False), lf, atol=1e-5), "letter features not scale-invariant"

    # v2 is only worth its complexity if it is genuinely rotation-invariant:
    # rotate the whole hand and the descriptor must barely move (v1 will not).
    hand = rng.random((21, 3)).astype(np.float32)
    theta = np.deg2rad(35.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]], dtype=np.float32)
    rotated = (hand - hand[0]) @ Rz.T + hand[0]
    v2, v2_rot = letter_features_v2(hand, False), letter_features_v2(rotated, False)
    assert v2.shape == (75,), f"v2 feature dim changed: {v2.shape}"
    drift = float(np.abs(v2 - v2_rot).max())
    assert drift < 1e-3, f"letter_features_v2 is not rotation-invariant (drift {drift})"
    # ...and the invariance is a real property, not a constant output
    assert np.abs(v2 - letter_features_v2(rng.random((21, 3)).astype(np.float32), False)).max() > 1e-2

    scaled = letter_features_v2(hand * 3.0, False)
    assert np.allclose(scaled, v2, atol=1e-4), "v2 not scale-invariant"

    aug = augment_hand(hand, np.random.default_rng(0))
    assert aug.shape == hand.shape and not np.allclose(aug, hand), "augment_hand did nothing"

    letter_model_path = MODEL_PATH.parent / "letter_rf.joblib"
    if letter_model_path.exists():
        bundle = joblib.load(letter_model_path)
        assert set(bundle) >= {"model", "features"}, "letter artifact missing feature version"
        dims = {"v1_screen": 40, "v2_palmframe": 75}[bundle["features"]]
        assert bundle["model"].n_features_in_ == dims, "letter model/feature dim mismatch"
        assert len(bundle["model"].classes_) == 26

    print("smoke ok")


if __name__ == "__main__":
    main()
