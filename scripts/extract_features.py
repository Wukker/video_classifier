#!/usr/bin/env python3
"""
Download YouTube videos and extract multimodal features:
  - Visual: CLIP ViT-B/32 embeddings (mean of N frames) → [512]
  - Audio:  MFCC mean + std (librosa)                   → [80]
  - Meta:   duration, bitrate, height, width, fps        → [5]

Output:
  data/embeddings/<video_id>.npz  — per-video arrays
  data/embeddings/manifest.csv    — video_id, category, status
"""

import argparse
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import open_clip
import yt_dlp
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
EMB_DIR = DATA_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_VIDEO = 5
AUDIO_SR = 16_000
N_MFCC = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Load CLIP once ────────────────────────────────────────────────────────────
def load_clip():
    print(f"Loading CLIP on {DEVICE}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess


# ── Feature extractors ────────────────────────────────────────────────────────
def extract_frames(video_path: Path, duration: float, frames_dir: Path) -> list[Path]:
    """Extract FRAMES_PER_VIDEO evenly-spaced frames via ffmpeg."""
    timestamps = [
        duration * i / (FRAMES_PER_VIDEO + 1)
        for i in range(1, FRAMES_PER_VIDEO + 1)
    ]
    paths = []
    for idx, ts in enumerate(timestamps):
        out = frames_dir / f"frame_{idx:02d}.jpg"
        result = subprocess.run(
            [
                "ffmpeg", "-ss", str(ts), "-i", str(video_path),
                "-vframes", "1", "-q:v", "2", str(out),
                "-loglevel", "error",
            ],
            capture_output=True,
        )
        if result.returncode == 0 and out.exists():
            paths.append(out)
    return paths


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract mono 16 kHz WAV via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-ar", str(AUDIO_SR), "-ac", "1",
            str(audio_path), "-loglevel", "error",
        ],
        capture_output=True,
    )
    return result.returncode == 0 and audio_path.exists()


def visual_embedding(frame_paths: list[Path], model, preprocess) -> np.ndarray:
    if not frame_paths:
        return np.zeros(512, dtype=np.float32)
    images = torch.stack(
        [preprocess(Image.open(p).convert("RGB")) for p in frame_paths]
    ).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(images)
    return emb.mean(dim=0).cpu().numpy().astype(np.float32)


def audio_embedding(audio_path: Path) -> np.ndarray:
    if not audio_path.exists():
        return np.zeros(N_MFCC * 2, dtype=np.float32)
    y, sr = librosa.load(str(audio_path), sr=AUDIO_SR, duration=60)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def meta_features(row: pd.Series) -> np.ndarray:
    return np.array(
        [row["duration"], row["bitrate"], row["height"], row["width"], row["frame rate"]],
        dtype=np.float32,
    )


# ── Download helper ───────────────────────────────────────────────────────────
def download_video(url: str, out_path: Path) -> bool:
    ydl_opts = {
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio/best[height<=480]/best",
        "outtmpl": str(out_path.with_suffix("")),  # yt-dlp adds extension
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # yt-dlp may have written <stem>.mp4 or similar
        candidate = out_path.parent / (out_path.stem + ".mp4")
        if candidate.exists() and candidate != out_path:
            candidate.rename(out_path)
        return out_path.exists()
    except Exception as e:
        print(f"    yt-dlp error: {e}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────
def main(n: int, seed: int, delay: float = 3.0):
    df = pd.read_csv(DATA_DIR / "youtube_data.csv")
    df = df[df["duration"] <= 60].reset_index(drop=True)
    print(f"Видео ≤60 сек в датасете: {len(df)}")
    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)

    model, preprocess = load_clip()
    records = []

    for idx, row in sample.iterrows():
        video_id = row["video_id"]
        url = row["url"]
        category = row["category"]
        out_npz = EMB_DIR / f"{video_id}.npz"

        print(f"\n[{idx + 1}/{n}] {video_id}  ({category})")

        if out_npz.exists():
            print("  → already extracted, skipping")
            records.append({"video_id": video_id, "category": category, "status": "skipped"})
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = tmp / f"{video_id}.mp4"
            audio_path = tmp / f"{video_id}.wav"
            frames_dir = tmp / "frames"
            frames_dir.mkdir()

            # 1. Download
            if not download_video(url, video_path):
                print("  ✗ download failed")
                records.append({"video_id": video_id, "category": category, "status": "failed"})
                continue
            print("  ✓ downloaded")

            # 2. Frames
            frame_paths = extract_frames(video_path, row["duration"], frames_dir)
            print(f"  ✓ {len(frame_paths)} frames extracted")

            # 3. Audio
            ok_audio = extract_audio(video_path, audio_path)
            print(f"  {'✓' if ok_audio else '✗'} audio extracted")

            # 4. Embeddings
            vis = visual_embedding(frame_paths, model, preprocess)
            aud = audio_embedding(audio_path)
            meta = meta_features(row)

            np.savez_compressed(out_npz, visual=vis, audio=aud, meta=meta)
            print(f"  ✓ saved → {out_npz.name}")

        records.append({"video_id": video_id, "category": category, "status": "ok"})

        pause = delay + random.uniform(0, delay / 2)
        print(f"  ⏳ пауза {pause:.1f}с...")
        time.sleep(pause)

    # Save manifest
    manifest = pd.DataFrame(records)
    manifest_path = EMB_DIR / "manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest = pd.concat([old, manifest]).drop_duplicates("video_id")
    manifest.to_csv(manifest_path, index=False)

    ok = (manifest["status"] == "ok").sum()
    print(f"\nDone: {ok}/{len(records)} videos processed successfully.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of videos to process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=3.0, help="Base pause between downloads (sec)")
    args = parser.parse_args()
    main(args.n, args.seed, args.delay)
