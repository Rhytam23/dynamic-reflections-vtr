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


def generate_descriptions(batches: Sequence[Sequence[str]], model_name: str = "google/gemma-2-2b-it",
                          max_new_tokens: int = 80) -> list:
    """For each list of retrieved captions, merge them into one description with a local LLM (text only).

    The model is loaded once for all batches.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # T4 has no native bfloat16 and Gemma-2 overflows in float16, so use float32 there (2B = ~10 GB).
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if bf16 else torch.float32, device_map="auto")
    outputs = []
    for retrieved in batches:
        chat = [{"role": "user", "content": build_fusion_prompt(retrieved)}]
        enc = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        outputs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
    del model
    torch.cuda.empty_cache()
    return outputs


def generate_description(retrieved: Sequence[str], **kw) -> str:
    return generate_descriptions([retrieved], **kw)[0]
