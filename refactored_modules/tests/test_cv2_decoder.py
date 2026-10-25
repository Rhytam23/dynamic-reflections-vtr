import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")

from refactored_modules.cv2_decoder import VideoDecoder


def _make_video(path, n=30, size=(64, 48)):
    """Video whose frame i is a flat colour encoding i (survives lossy compression well enough)."""
    mp4 = path.with_suffix(".mp4")  # cv2 picks the container from the extension
    w = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    assert w.isOpened()
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 8, 255 - i * 8, 100), np.uint8))
    w.release()
    mp4.rename(path)


def test_decoder_matches_authors_usage(tmp_path):
    p = tmp_path / "clip.bin"  # PVD stores mp4 bytes with a .bin extension
    _make_video(p)
    v = VideoDecoder(p, dimension_order="NHWC")
    assert len(v) in range(29, 32)
    frames = v.get_frames_at([0, 10, 25]).data.numpy()
    assert frames.shape == (3, 48, 64, 3) and frames.dtype == np.uint8
    # cv2 wrote BGR (i*8, 255-i*8, 100): in the decoded RGB frames R stays ~100 and B grows with i
    px = frames[:, 24, 32].astype(int)
    assert np.all(np.abs(px[:, 0] - 100) < 20)
    assert px[0, 2] < px[1, 2] < px[2, 2]


def test_unsorted_duplicate_and_out_of_range_indices(tmp_path):
    p = tmp_path / "clip.bin"
    _make_video(p, n=12)
    out = VideoDecoder(p).get_frames_at([5, 5, 2, 999]).data.numpy()
    assert out.shape[0] == 4
    assert np.array_equal(out[0], out[1])       # duplicate index -> same frame
    assert np.array_equal(out[3], VideoDecoder(p).get_frames_at([11]).data.numpy()[0])  # clamped to last frame


def test_missing_file_raises(tmp_path):
    with pytest.raises(IOError):
        VideoDecoder(tmp_path / "nope.bin")
