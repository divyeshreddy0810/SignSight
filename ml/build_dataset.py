"""
Builds the labeled landmark dataset for the 8-word vocabulary from WLASL.

For each gloss instance: downloads the source YouTube video (cached by video_id
so instances sharing a source video only download once), extracts the labeled
frame range, runs MediaPipe Holistic per-frame to get pose + both hands, then
resamples the sequence to a fixed-length window matching the live frontend
(frontend/index.html's WINDOW_SIZE). Output: one .npy landmark array per clip
plus a labels.csv manifest. Failures (dead YouTube links, download errors) are
logged separately rather than silently dropped, since link rot is a known,
citable property of WLASL.

Usage: python ml/build_dataset.py
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUBSET_JSON = ROOT / "data" / "wlasl_subset.json"
RAW_VIDEOS_DIR = ROOT / "data" / "raw_videos"
FRAMES_DIR = ROOT / "data" / "frames_tmp"
PROCESSED_DIR = ROOT / "data" / "processed"
LABELS_CSV = PROCESSED_DIR / "labels.csv"
FAILURES_CSV = PROCESSED_DIR / "failures.csv"

WINDOW_SIZE = 30  # must match frontend/index.html WINDOW_SIZE
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_KEYPOINTS = NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS  # 75

mp_holistic = mp.solutions.holistic


def download_video(video_id: str, url: str) -> Path | None:
    dest = RAW_VIDEOS_DIR / f"{video_id}.mp4"
    if dest.exists():
        return dest
    result = subprocess.run(
        ["yt-dlp", "-f", "best[ext=mp4]/best", "-o", str(dest), url],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0 or not dest.exists():
        return None
    return dest


def video_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def extract_frames(video_path: Path, frame_start: int, frame_end: int, out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # WLASL uses frame_end == -1 as a sentinel meaning "to the end of the video".
    if frame_end == -1:
        frame_end = video_frame_count(video_path) - 1
        if frame_end < frame_start:
            return 0
    select_expr = f"between(n\\,{frame_start}\\,{frame_end})"
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vf", f"select='{select_expr}'", "-vsync", "0",
         str(out_dir / "frame_%04d.png")],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return 0
    return len(list(out_dir.glob("*.png")))


def landmarks_for_clip(holistic, frame_paths: list[Path]) -> np.ndarray:
    """Returns shape (num_frames, NUM_KEYPOINTS, 4) -> [x, y, z, visibility]."""
    sequence = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_rgb)

        pose = (
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            if results.pose_landmarks else
            [[0, 0, 0, 0]] * NUM_POSE_LANDMARKS
        )

        def hand(landmarks):
            if landmarks:
                return [[lm.x, lm.y, lm.z, 1.0] for lm in landmarks.landmark]
            return [[0, 0, 0, 0]] * NUM_HAND_LANDMARKS

        frame_kp = pose + hand(results.left_hand_landmarks) + hand(results.right_hand_landmarks)
        sequence.append(frame_kp)
    return np.array(sequence, dtype=np.float32)


def resample_to_window(sequence: np.ndarray, window_size: int) -> np.ndarray:
    """Linearly resamples a (T, K, 4) sequence to (window_size, K, 4) along time."""
    t_orig = sequence.shape[0]
    if t_orig == window_size:
        return sequence
    orig_idx = np.linspace(0, t_orig - 1, t_orig)
    target_idx = np.linspace(0, t_orig - 1, window_size)
    resampled = np.empty((window_size, *sequence.shape[1:]), dtype=np.float32)
    for k in range(sequence.shape[1]):
        for c in range(sequence.shape[2]):
            resampled[:, k, c] = np.interp(target_idx, orig_idx, sequence[:, k, c])
    return resampled


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUBSET_JSON) as f:
        subset = json.load(f)

    labels_rows = []
    failure_rows = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for entry in subset:
            gloss = entry["gloss"].strip().lower()
            for inst in entry["instances"]:
                clip_id = f"{gloss.replace(' ', '_')}_{inst['video_id']}_{inst['instance_id']}"
                out_npy = PROCESSED_DIR / f"{clip_id}.npy"
                if out_npy.exists():
                    continue  # resumable: skip already-processed clips

                print(f"[{clip_id}] downloading {inst['url']} ...", flush=True)
                video_path = download_video(inst["video_id"], inst["url"])
                def failure_row(reason):
                    return {
                        "gloss": gloss, "video_id": inst["video_id"], "instance_id": inst["instance_id"],
                        "url": inst["url"], "frame_start": inst["frame_start"], "frame_end": inst["frame_end"],
                        "signer_id": inst["signer_id"], "split": inst["split"], "reason": reason,
                    }

                if video_path is None:
                    print(f"[{clip_id}] FAILED: download error", flush=True)
                    failure_rows.append(failure_row("download_failed"))
                    continue

                n_frames = extract_frames(video_path, inst["frame_start"], inst["frame_end"], FRAMES_DIR)
                if n_frames == 0:
                    print(f"[{clip_id}] FAILED: frame extraction produced 0 frames", flush=True)
                    failure_rows.append(failure_row("frame_extraction_failed"))
                    continue

                frame_paths = sorted(FRAMES_DIR.glob("*.png"))
                sequence = landmarks_for_clip(holistic, frame_paths)
                windowed = resample_to_window(sequence, WINDOW_SIZE)
                np.save(out_npy, windowed)

                labels_rows.append({
                    "clip_id": clip_id,
                    "gloss": gloss,
                    "split": inst["split"],
                    "signer_id": inst["signer_id"],
                    "video_id": inst["video_id"],
                    "source_url": inst["url"],
                    "raw_frame_count": n_frames,
                })
                print(f"[{clip_id}] OK ({n_frames} raw frames -> {WINDOW_SIZE}-frame window)", flush=True)

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)

    # labels.csv appends (resumable runs only ever add new clips).
    if labels_rows:
        file_exists = LABELS_CSV.exists()
        with open(LABELS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "clip_id", "gloss", "split", "signer_id", "video_id", "source_url", "raw_frame_count"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(labels_rows)

    # failures.csv is rebuilt from scratch every run: an instance is failed iff
    # its clip .npy is absent, so reruns never duplicate rows (the old append
    # behaviour re-logged every dead URL on each resume). Reasons carry over
    # from the previous manifest for instances not attempted this run.
    prior_reasons = {}
    if FAILURES_CSV.exists():
        with open(FAILURES_CSV, newline="") as f:
            for row in csv.DictReader(f):
                prior_reasons[(str(row["video_id"]), str(row["instance_id"]))] = row["reason"]
    run_reasons = {(str(r["video_id"]), str(r["instance_id"])): r["reason"] for r in failure_rows}

    all_failures = []
    for entry in subset:
        gloss = entry["gloss"].strip().lower()
        for inst in entry["instances"]:
            clip_id = f"{gloss.replace(' ', '_')}_{inst['video_id']}_{inst['instance_id']}"
            if (PROCESSED_DIR / f"{clip_id}.npy").exists():
                continue
            key = (str(inst["video_id"]), str(inst["instance_id"]))
            all_failures.append({
                "gloss": gloss, "video_id": inst["video_id"], "instance_id": inst["instance_id"],
                "url": inst["url"], "frame_start": inst["frame_start"], "frame_end": inst["frame_end"],
                "signer_id": inst["signer_id"], "split": inst["split"],
                "reason": run_reasons.get(key, prior_reasons.get(key, "download_failed")),
            })
    with open(FAILURES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gloss", "video_id", "instance_id", "url", "frame_start", "frame_end", "signer_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(all_failures)

    print(f"\nDone. {len(labels_rows)} clips processed this run, {len(failure_rows)} failures this run "
          f"({len(all_failures)} total unretrieved).")
    print(f"Manifests: {LABELS_CSV}, {FAILURES_CSV}")


if __name__ == "__main__":
    main()
