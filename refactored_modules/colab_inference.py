"""Colab-only helpers for the video-to-text demo (need a GPU, transformers, cv2).

Not covered by the local tests: they load Gemma-2-2B-it and decode video files.
"""
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .retrieval import build_fusion_prompt

logger = logging.getLogger(__name__)


def sample_frames(video_path: Path, n: int = 5):
    """n evenly spaced RGB frames from a video file (the PVD '.bin' files are mp4)."""
    import cv2
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in np.linspace(0, max(total - 1, 0), n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def generate_description(retrieved: Sequence[str], model_name: str = "google/gemma-2-2b-it", max_new_tokens: int = 80) -> str:
    """Merge retrieved captions into one description with a local LLM (text only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    chat = [{"role": "user", "content": build_fusion_prompt(retrieved)}]
    ids = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    del model
    torch.cuda.empty_cache()
    return text
