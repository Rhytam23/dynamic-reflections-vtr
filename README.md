# Dynamic Reflections: video-text alignment on Colab

Course project based on **"DynamicReflections: Probing Video Representations with Text Alignment"**
(Zhu et al., ICLR 2026, [arXiv 2511.02767](https://arxiv.org/abs/2511.02767); paper in `paper.pdf`,
authors' code: [google-deepmind/platonic_rep_video](https://github.com/google-deepmind/platonic_rep_video)).

The paper measures how well video encoders line up with text encoders using **mutual k-NN alignment** and shows the
score grows with the number of frames and captions given at test time. This repo reproduces that at small scale and
adds a video-to-text retrieval step.

## Constraints and what to expect
- Runs on **Colab with a T4 GPU (16 GB)**, so the text model is **Gemma-2-2B-it**, not the paper's Gemma-2-9B.
  Absolute scores will be lower than the paper's (about 0.2-0.4); we compare **trends**, not exact values.
- Scores depend on N and k (chance level is about k/N), so every result carries N, k and its chance level.
- The `0.85` that used to be returned by the pipeline was a hardcoded placeholder. It has been removed; scores are measured.

## Run it (one go)
1. In Colab, add a secret **`HF_TOKEN`** (key icon, enable notebook access) and accept the license at
   [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) with the same account.
2. Open `colab_run_pfinal.ipynb` from this repo, set the runtime to **T4 GPU**, and choose **Runtime > Run all**.
   The only interaction is the Google Drive permission pop-up.
3. To save GPU units, run it once on a **CPU runtime** first: that only downloads the videos to Drive. Then switch to T4 and Run all.

The notebook is built to fail early and cheaply:
- **Preflight** (Hugging Face access, disk, GPU) runs before anything expensive.
- A **smoke test on 8 videos** exercises every model and code path before the long extraction.
- Every step is a **stage**: a failed stage is recorded and only the stages that depend on it are skipped;
  a critical failure stops the run. All stage statuses are printed at the end and saved to Drive.
- Extraction is **resumable** (finished feature files are skipped) and everything is stored on Drive.
- The GPU runtime is **released automatically** when the run ends or fails (`DISCONNECT_WHEN_DONE`).

## What is implemented (`refactored_modules/`)
| Module | Purpose |
|---|---|
| `alignment.py` | Mutual k-NN (paper Eq. 1) with the authors' outlier clamp and layer x layer sweep; checked against a transcription of their algorithm |
| `features.py` | Wraps the authors' extraction scripts; overrides the model list (their `pvd_sample` config also runs the 9B); feature-file naming (identical to theirs) |
| `pipeline.py` | Resumable dataset download and the real baseline alignment score |
| `sweeps.py` | Frames x captions grid, encoder comparison table |
| `scaling_law.py` | Fit of Eq. 2, `S_inf - (C_f n_f^-a + C_c n_c^-b)`, with R^2 |
| `plots.py` | Layer heatmap, sweep curves, fit quality |
| `retrieval.py` | Video -> caption retrieval (ridge map or relative representations), recall@k on held-out videos |
| `colab_inference.py` | Colab-only: frame preview and an LLM that merges retrieved captions |
| `preflight.py`, `runner.py` | Fail-fast checks; stage runner with dependencies and auto-disconnect |
| `cv2_decoder.py` | OpenCV replacement for `torchcodec` (see below) |
| `utils.py`, `config.py` | Environment setup and patches for the authors' repo; settings |

**Patches applied to the authors' repo** (`utils.patch_scripts`, idempotent, fail loudly if their code changed):
- `torchcodec` is replaced by OpenCV decoding. Its CUDA build failed to load on Colab (`libnppicc.so.12`), and the authors import it even for text-only extraction.
- The dataset download is resumable and works with current `datasets`.
- DataLoader workers are capped at 4 (10 can exhaust a 12 GB Colab machine).
- A timm config fix for some encoders.

**Video-to-text note:** the video and text spaces have different sizes, so retrieval needs paired training examples
(it is not zero-shot). Layers are chosen on the training split only; recall@k is reported on held-out videos.
The demo uses held-out dataset videos; encoding an arbitrary new video is not implemented.

## Tests (no GPU needed)
```bash
pip install pytest numpy scipy matplotlib torch opencv-python-headless pillow
python -m pytest refactored_modules/tests -q
```
41 tests, including: the metric against the authors' algorithm, the scaling-law fit recovering the paper's parameters,
the OpenCV decoder driving the authors' real `PerceptionVideoDataset`, feature filenames matching the authors' code, and a
**simulation of the whole notebook** (with Colab, network and GPU faked) covering success, a failed Hugging Face check,
a CPU-only runtime and a failing encoder.

## Status
- Tested locally: all code above. Also checked against the authors' real code: their CLI parser accepts every command we issue,
  and their dataset class runs on our decoder.
- **Not yet run on Colab**: model loading and feature extraction on the real GPU, and all real numbers.
  Results table: to be filled in after the Colab run (nothing is reported here until then).

## Credits
Method and data pipeline by Zhu, Han, Guibas, Patraucean and Ovsjanikov; we use their released code for feature extraction.
