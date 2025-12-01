#!/usr/bin/env python3
"""
Extract uniformly sampled frames from PAIRPHOTO videos into an aligned image tree.

Design:
- Offline extraction BEFORE training (do not extract during the training loop).
- Uniform sampling: N frames per video (e.g., 120 for ~15s).
- Output layout mirrors input: dst/setXXXX/bitcode/{0001..NNNN}.jpg
- Assumes each leaf dir contains one video (e.g., .MOV/.MP4). If multiple, picks the first.
- Color/style jitter is optional at extraction time (pair-consistent index alignment still holds).
- Optional 180° rotation to fix upside-down videos (since OpenCV ignores orientation metadata).
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional
import random

import numpy as np
from PIL import Image, ImageEnhance

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def find_leaf_video_dirs(src_root: Path) -> List[Path]:
    leaf_dirs: List[Path] = []
    for set_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        for bit_dir in sorted([p for p in set_dir.iterdir() if p.is_dir()]):
            leaf_dirs.append(bit_dir)
    return leaf_dirs


def pick_first_video_file(leaf_dir: Path) -> Optional[Path]:
    exts = {".mov", ".mp4", ".m4v", ".avi", ".hevc", ".mkv"}
    files = sorted([p for p in leaf_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    return files[0] if files else None


def uniform_indices(total_frames: int, N: int) -> List[int]:
    # Sample indices at centers of sub-intervals; clamp to valid range
    if total_frames <= 0 or N <= 0:
        return []
    idxs = []
    for i in range(N):
        # position in [0, total_frames-1]
        pos = int(round((i + 0.5) / N * total_frames)) - 1
        if pos < 0:
            pos = 0
        if pos >= total_frames:
            pos = total_frames - 1
        idxs.append(pos)
    # Deduplicate while preserving order if total_frames is small
    dedup = []
    seen = set()
    for k in idxs:
        if k not in seen:
            dedup.append(k)
            seen.add(k)
    # If dedup smaller than N, we pad by repeating last frame index
    while len(dedup) < N and dedup:
        dedup.append(dedup[-1])
    return dedup[:N]


def _apply_style_bgr(frame_bgr: np.ndarray, seed: int) -> np.ndarray:
    """Apply a deterministic light color jitter in RGB space, then return BGR."""
    if cv2 is None:
        return frame_bgr
    rnd = random.Random(seed)
    bright = 1.0 + rnd.uniform(-0.18, 0.18)
    contrast = 1.0 + rnd.uniform(-0.20, 0.20)
    saturation = 1.0 + rnd.uniform(-0.18, 0.18)
    wb = [1.0 + rnd.uniform(-0.06, 0.06) for _ in range(3)]
    gamma = 1.0 + rnd.uniform(-0.10, 0.10)

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    img = ImageEnhance.Brightness(img).enhance(bright)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    # white-balance like channel scaling
    arr = np.asarray(img).astype(np.float32)
    arr[..., 0] *= wb[0]
    arr[..., 1] *= wb[1]
    arr[..., 2] *= wb[2]
    arr = np.clip(arr, 0, 255)
    # gamma
    arr = 255.0 * np.power(arr / 255.0, 1.0 / gamma)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def extract_with_opencv(video_path: Path, out_dir: Path, N: int, overwrite: bool = False,
                        rotate180: bool = False, style_jitter: bool = False) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        print(f"[ERROR] OpenCV not available. Please `pip install opencv-python-headless`.")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[WARN] Total frames unknown/zero for {video_path}, attempting to read sequentially.")
        # Try to read sequentially to estimate length
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        total_frames = len(frames)
        if total_frames == 0:
            print(f"[ERROR] Could not read frames from {video_path}")
            return False
        idxs = uniform_indices(total_frames, N)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(idxs, start=1):
            frame = frames[idx]
            out_path = out_dir / f"{i:04d}.jpg"
            if out_path.exists() and not overwrite:
                continue
            if rotate180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            if style_jitter:
                seed = hash((video_path.name, idx)) & 0xFFFFFFFF
                frame = _apply_style_bgr(frame, seed)
            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                print(f"[WARN] Failed to write {out_path}")
        return True

    idxs = uniform_indices(total_frames, N)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(idxs, start=1):
        out_path = out_dir / f"{i:04d}.jpg"
        if out_path.exists() and not overwrite:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] Could not grab frame {idx} from {video_path}")
            continue
        if rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if style_jitter:
            seed = hash((video_path.name, idx)) & 0xFFFFFFFF
            frame = _apply_style_bgr(frame, seed)
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"[WARN] Failed to write {out_path}")
    cap.release()
    return True


def main():
    ap = argparse.ArgumentParser(description="Extract uniformly sampled frames from PAIRPHOTO videos.")
    ap.add_argument("--src", type=str, required=True, help="PAIRPHOTO root, e.g., /Users/karl/Downloads/PAIRPHOTO")
    ap.add_argument("--dst", type=str, default="external/pairphoto_frames", help="Output root for frames")
    ap.add_argument("--frames-per-video", type=int, default=120, help="Number of frames per video (uniformly sampled)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    ap.add_argument("--rotate-180", action="store_true", help="Rotate each frame by 180 degrees before saving")
    ap.add_argument("--style-jitter", action="store_true", help="Apply deterministic per-frame color jitter at extraction")
    args = ap.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    N = int(args.frames_per_video)

    if not src_root.exists():
        print(f"[ERROR] src root not found: {src_root}")
        raise SystemExit(1)
    dst_root.mkdir(parents=True, exist_ok=True)

    leaf_dirs = find_leaf_video_dirs(src_root)
    if not leaf_dirs:
        print(f"[WARN] No set/bitcode subfolders found in {src_root}")

    total = 0
    ok_cnt = 0
    for leaf in leaf_dirs:
        video = pick_first_video_file(leaf)
        if not video:
            print(f"[SKIP] No video found in {leaf}")
            continue
        # dst path mirrors src leaf relative to src_root
        rel = leaf.relative_to(src_root)
        out_dir = dst_root / rel
        total += 1
        print(f"[INFO] Extracting {video.name} -> {out_dir} (N={N})")
        if extract_with_opencv(video, out_dir, N=N, overwrite=args.overwrite,
                               rotate180=args.rotate_180, style_jitter=args.style_jitter):
            ok_cnt += 1
    print(f"[DONE] Processed {total}, succeeded {ok_cnt}. Output: {dst_root}")


if __name__ == "__main__":
    main()
