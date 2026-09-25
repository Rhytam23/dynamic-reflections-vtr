"""OpenCV stand-in for `torchcodec.decoders.VideoDecoder` (only the parts the authors use).

torchcodec's CUDA build fails to load on Colab (missing libnppicc.so.12), and their code
imports it even for text-only extraction. This file is copied into the authors' package by
`patch_scripts` as `vprh/dataloaders/_cv2_decoder.py`, so it must stay self-contained.

API used by `vprh.dataloaders.pvd`:
    video = VideoDecoder(path, dimension_order='NHWC')
    len(video)
    video.get_frames_at(indices).data   # uint8 tensor [N, H, W, 3], RGB
"""
import cv2
import numpy as np
import torch


class _Frames:
    def __init__(self, data):
        self.data = data


class VideoDecoder:
    def __init__(self, path, dimension_order="NHWC", **_):
        if dimension_order != "NHWC":
            raise NotImplementedError("only NHWC is supported")
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.path}")
            self._n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def __len__(self):
        return self._n

    def _decode(self, wanted):
        """One sequential pass (exact indices; cv2 seeking is unreliable). Returns ({index: RGB frame}, frames read)."""
        cap = cv2.VideoCapture(self.path)
        got, pos = {}, 0
        try:
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {self.path}")
            targets = set(wanted)
            stop = max(wanted) if wanted else -1
            while pos <= stop and cap.grab():
                if pos in targets:
                    ok, frame = cap.retrieve()
                    if ok:
                        got[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos += 1
            # if grab() failed early, pos is the real frame count; otherwise the count is >= pos
            return got, pos
        finally:
            cap.release()

    def get_frames_at(self, indices):
        """Frames at `indices` (any order, duplicates allowed) as a uint8 [N, H, W, 3] RGB tensor.

        Indices past the last decodable frame (the reported frame count can be off) reuse the last frame.
        """
        idx = [int(i) for i in indices]
        got, real_count = self._decode(sorted(set(idx)))
        missing = [i for i in set(idx) if i not in got]
        if missing:  # video ended before these indices: reuse its true last frame
            if real_count == 0:
                raise IOError(f"No decodable frames in {self.path}")
            last, _ = self._decode([real_count - 1])
            for i in missing:
                got[i] = last[real_count - 1]
        return _Frames(torch.from_numpy(np.stack([got[i] for i in idx])))
