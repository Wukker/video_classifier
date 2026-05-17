#!/usr/bin/env python3
"""
Video genre classifier.
Usage: python predict.py <youtube_url>
"""
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import cv2

MODELS_DIR   = Path(__file__).parent.parent / "models"
SCRIPTS_DIR  = Path(__file__).parent

sys.path.insert(0, str(SCRIPTS_DIR))
import feature_extractor as yt8m_fe


# ── Модель ────────────────────────────────────────────────────────────────────

class FrameRNN(nn.Module):
    def __init__(self, dim_rgb, dim_aud, n_classes,
                 proj_dim=256, rnn_hidden=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_rgb + dim_aud, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(proj_dim, rnn_hidden, batch_first=True, bidirectional=True)
        out_dim = rnn_hidden * 2
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, rgb, aud, lengths):
        x = self.proj(torch.cat([rgb, aud], dim=-1))
        lens_cpu = lengths.detach().cpu().clamp(min=1)
        packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h)


# ── Загрузка модели и конфига ─────────────────────────────────────────────────

def load_model():
    cfg  = json.load(open(MODELS_DIR / "config.json"))
    norm = np.load(MODELS_DIR / "norm_stats.npz")

    model = FrameRNN(cfg["dim_rgb"], cfg["dim_audio"], cfg["n_classes"])
    model.load_state_dict(torch.load(MODELS_DIR / "best_rnn.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()
    return model, cfg, norm


# ── Скачивание видео ──────────────────────────────────────────────────────────

def download_video(url: str, out_dir: Path) -> Path:
    js_runtime = None
    for name in ("node", "deno", "bun"):
        p = shutil.which(name)
        if p:
            js_runtime = f"{name}:{p}"
            break

    cmd = [
        "yt-dlp",
        "-f", "bv*[vcodec^=avc1][height<=720]+ba/b[height<=720]",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "--no-playlist", "--no-progress",
        "--download-sections", "*0-60",
        "--print", "after_move:filepath",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        url,
    ]
    if js_runtime:
        cmd += ["--js-runtimes", js_runtime]

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
    return Path(lines[-1])


# ── Извлечение признаков ──────────────────────────────────────────────────────

def extract_features(video_path: Path, extractor, L: int, dim_aud: int):
    cap = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps)), 1)

    rgb_feats = []
    idx = 0
    while len(rgb_feats) < L:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            feat = extractor.extract_rgb_frame_features(frame[:, :, ::-1], apply_pca=True)
            rgb_feats.append(feat.astype(np.float32))
        idx += 1
    cap.release()

    if not rgb_feats:
        raise RuntimeError(f"Не удалось извлечь кадры из {video_path}")

    T       = min(len(rgb_feats), L)
    rgb_arr = np.stack(rgb_feats[:T], axis=0)
    aud_arr = np.zeros((T, dim_aud), dtype=np.float32)
    return rgb_arr, aud_arr, T


# ── Предсказание ──────────────────────────────────────────────────────────────

def predict(url: str):
    print("Загрузка модели...")
    model, cfg, norm = load_model()

    L       = cfg["L"]
    DIM_RGB = cfg["dim_rgb"]
    DIM_AUD = cfg["dim_audio"]
    GENRES  = cfg["genres"]

    print("Инициализация экстрактора признаков (первый раз ~100 MB)...")
    extractor = yt8m_fe.YouTube8MFeatureExtractor()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Скачивание видео: {url}")
        video_path = download_video(url, Path(tmp))
        print(f"Извлечение признаков...")
        rgb_arr, aud_arr, T = extract_features(video_path, extractor, L, DIM_AUD)

    print(f"Кадров извлечено: {T}")

    rgb = np.zeros((L, DIM_RGB), dtype=np.float32)
    aud = np.zeros((L, DIM_AUD), dtype=np.float32)
    rgb[:T] = (rgb_arr - norm["rgb_mean"]) / norm["rgb_std"]
    aud[:T] = (aud_arr - norm["aud_mean"]) / norm["aud_std"]

    rgb_t = torch.from_numpy(rgb).unsqueeze(0)
    aud_t = torch.from_numpy(aud).unsqueeze(0)
    len_t = torch.tensor([T], dtype=torch.long)

    with torch.no_grad():
        logits = model(rgb_t, aud_t, len_t)
        probs  = torch.softmax(logits, dim=-1)[0].numpy()

    top = np.argsort(probs)[::-1]

    print("\n" + "─" * 35)
    print(f"  Результат классификации")
    print("─" * 35)
    for i in top[:5]:
        bar = "█" * int(probs[i] * 30)
        print(f"  {GENRES[i]:<14} {probs[i]*100:>5.1f}%  {bar}")
    print("─" * 35)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <youtube_url>")
        sys.exit(1)
    predict(sys.argv[1])
