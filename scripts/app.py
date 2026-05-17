#!/usr/bin/env python3
import sys
import tempfile
import numpy as np
import torch
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import feature_extractor as yt8m_fe
from predict import load_model, download_video, extract_features

print("Loading model...")
model, cfg, norm = load_model()
print("Initializing feature extractor (~100 MB on first run)...")
extractor = yt8m_fe.YouTube8MFeatureExtractor()
print("Ready.")

L       = cfg["L"]
DIM_RGB = cfg["dim_rgb"]
DIM_AUD = cfg["dim_audio"]
GENRES  = cfg["genres"]


def classify(url: str):
    url = (url or "").strip()
    if not url:
        return {}

    with tempfile.TemporaryDirectory() as tmp:
        video_path = download_video(url, Path(tmp))
        rgb_arr, aud_arr, T = extract_features(video_path, extractor, L, DIM_AUD)

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

    return {GENRES[i]: float(probs[i]) for i in range(len(GENRES))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="YouTube URL",
        placeholder="https://www.youtube.com/shorts/...",
        lines=1,
    ),
    outputs=gr.Label(num_top_classes=5, label="Genre"),
    title="Video Genre Classifier",
    description="Paste a YouTube or Shorts link. Prediction takes 1–3 min on CPU.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
