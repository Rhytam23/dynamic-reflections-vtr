# Dynamic Reflections: video-text alignment on free Colab

Course project based on **"DynamicReflections: Probing Video Representations with Text Alignment"**
(Zhu et al., ICLR 2026, [arXiv 2511.02767](https://arxiv.org/abs/2511.02767); paper in `paper.pdf`,
authors' code: [google-deepmind/platonic_rep_video](https://github.com/google-deepmind/platonic_rep_video)).

The paper measures how well video encoders line up with text encoders using **mutual k-NN alignment** and shows the
score grows with the number of frames and captions given at test time. This repo reproduces that at small scale and
adds a video-to-text retrieval step.

## Constraints and what to expect
- Runs on **free Colab (T4, 16 GB)**, so the text model is **Gemma-2-2B-it**, not the paper's Gemma-2-9B.
  Absolute scores will be lower than the paper's (about 0.2-0.4); we compare **trends**, not exact values.
- Scores depend on N and k (chance level is about k/N), so every result carries N, k and its chance level.
- The `0.85` that used to be returned by the pipeline was a hardcoded placeholder. It has been removed; scores are measured.

## What is implemented (`refactored_modules/`)
| Module | Purpose |
|---|---|
| `alignment.py` | Mutual k-NN (paper Eq. 1) with the authors' outlier clamp and layer x layer sweep; checked against a transcription of their algorithm |
| `features.py` | Wraps the authors' extraction scripts; overrides the model list (their `pvd_sample` config also runs the 9B); feature-file naming |
| `pipeline.py` | Dataset download and the real baseline alignment score |
| `sweeps.py` | Frames x captions grid, encoder comparison table |
| `scaling_law.py` | Fit of Eq. 2, `S_inf - (C_f n_f^-a + C_c n_c^-b)`, with R^2 |
| `plots.py` | Layer heatmap, sweep curves, fit quality |
| `retrieval.py` | Video -> caption retrieval (ridge map or relative representations), recall@k on held-out videos |
| `colab_inference.py` | Colab-only: frame preview and an LLM that merges retrieved captions |
| `utils.py`, `config.py` | Environment setup and patches for the authors' repo; settings |

**Video-to-text note:** the video and text spaces have different sizes, so retrieval needs paired training examples
(it is not zero-shot). Layers are chosen on the training split only; recall@k is reported on held-out videos.
The demo uses held-out dataset videos; encoding an arbitrary new video is not implemented.

## Run it
1. Open `colab_run_pfinal.ipynb` in Colab with a **T4 GPU** and a Hugging Face token that has accepted the Gemma license.
2. Run top to bottom. Section 3 gives the first real score; section 4 (sweeps) is the slow part, and extraction
   resumes after a disconnect because finished feature files are skipped.

Local tests (no GPU or torch needed, synthetic data):
```bash
pip install pytest numpy scipy matplotlib
python -m pytest refactored_modules/tests -q
```

## Status
- Done and tested locally (20 tests): metric, scaling-law fit, sweep/grid logic, plotting, retrieval, extraction command construction.
- **Not yet run on Colab**: feature extraction and all real numbers. Untested locally: `colab_inference.py`, the notebook itself,
  and how the authors' extractor behaves for each frame count (DINOv2-large uses 8-frame clips, VideoMAEv2 16).
- Results table: to be filled in after the Colab run (nothing is reported here until then).

## Credits
Method and data pipeline by Zhu, Han, Guibas, Patraucean and Ovsjanikov; we use their released code for feature extraction.
