"""Scaled-down CPU run of the DynamicReflections pipeline (no Colab, no GPU).

Uses this repo's own modules around the authors' code. Everything reported is computed
from features extracted in this run; nothing is estimated or filled in by hand.
Usage: python run_cpu_scaled.py [--n 150] [--k 10] [--llm bloom-560m] [--stage all]
"""
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from refactored_modules import plots, retrieval, sweeps
from refactored_modules.features import extract_features, feature_path, load_features
from refactored_modules.pipeline import run_alignment_pipeline
from refactored_modules.runner import StageRunner
from refactored_modules.utils import clone_authors_repo, patch_scripts

logging.basicConfig(level=logging.INFO, format="%(message)s")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, default=Path("/home/user/work/platonic_rep_video"))
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--llm", default="bloom-560m")  # ungated; gemma2-2b-it needs HF_TOKEN
ap.add_argument("--vision", default="dinov2_large_video")
ap.add_argument("--frames", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--captions", type=int, nargs="+", default=[1, 2, 4, 10])
ap.add_argument("--out", type=Path, default=Path("results/ours"))
args = ap.parse_args()

repo = args.repo.resolve()
out = args.out.resolve()
feature_dir = repo / "results" / f"cpu_n{args.n}"
subset = repo / "assets" / f"subset_{args.n}.jsonl"
venv_py = repo / ".venv" / "bin" / "python"
hf_token = os.environ.get("HF_TOKEN")
llm = args.llm
if llm.startswith("gemma") and not hf_token:
    raise SystemExit("gated model requested but HF_TOKEN is not set")
meta = {
    "hardware": f"CPU only, {os.cpu_count()} cores, {platform.processor() or platform.machine()}, no GPU",
    "text_model": llm, "video_model": args.vision, "n": args.n, "k": args.k,
    "chance_level": args.k / (args.n - 1),
    "scale_down": f"first {args.n} of 1024 PVD videos; frames {args.frames}; captions {args.captions}; "
                  "single video model; NOT the full T4 experiment",
}
run = StageRunner()


@run.stage("setup", critical=True)
def _():
    clone_authors_repo(repo)
    patch_scripts(repo)
    rows = [json.loads(l) for l in (repo / "assets/pe_video_dataset_1k_rephrased.jsonl").open()][: args.n]
    subset.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


@run.stage("download_videos", critical=True, needs=["setup"])
def _():
    # the patched authors' download_pvd.py; run with the venv python that has `datasets`
    subprocess.run([str(venv_py), "src/vprh/misc/download_pvd.py", "-l", str(subset), "-o", "data/pvd"],
                   cwd=repo, check=True)


@run.stage("extract_text", critical=True, needs=["setup"])
def _():
    sweeps.extract_caption_sweep(repo, feature_dir, llm, args.captions, hf_token, annotation_path=subset)


@run.stage("extract_video", critical=True, needs=["download_videos"])
def _():
    sweeps.extract_frame_sweep(repo, feature_dir, args.vision, args.frames, annotation_path=subset)


@run.stage("grid", needs=["extract_text", "extract_video"])
def _():
    res = sweeps.fit_grid(sweeps.compute_grid(feature_dir, llm, args.vision, args.frames, args.captions, k=args.k))
    res["meta"] = meta
    sweeps.save_result(res, out / f"grid_{llm}_{args.vision}_n{args.n}.json")
    plots.plot_sweeps([res], out / f"sweeps_n{args.n}.png")
    if res["fit"]:
        plots.plot_fit(res, out / f"fit_n{args.n}.png")
    print(sweeps.comparison_table([res]))
    return res


@run.stage("layer_heatmap", needs=["extract_text", "extract_video"])
def _():
    nf, nc = max(args.frames), max(args.captions)
    # run_alignment_pipeline reads the unsuffixed files; use the suffixed ones via direct sweep
    from refactored_modules.alignment import layer_sweep
    v = load_features(feature_path(feature_dir, "pvd", args.vision, "cls", num_frames=nf))
    t = load_features(feature_path(feature_dir, "pvd", llm, "avg", num_captions=nc))
    scores, best = layer_sweep(v, t, k=args.k)
    plots.plot_layer_heatmap(scores, best, out / f"layers_n{args.n}.png")
    res = {"meta": meta, "n_frames": nf, "n_captions": nc, "best": best, "chance_level": args.k / (best["n"] - 1)}
    sweeps.save_result(res, out / f"layers_{llm}_{args.vision}_n{args.n}.json")
    return res


@run.stage("retrieval", needs=["extract_text", "extract_video"])
def _():
    nf, nc = max(args.frames), max(args.captions)
    v = load_features(feature_path(feature_dir, "pvd", args.vision, "cls", num_frames=nf))
    t = load_features(feature_path(feature_dir, "pvd", llm, "avg", num_captions=nc))
    r = retrieval.evaluate_retrieval(v, t, k_align=args.k)
    r.pop("_sims")
    r["meta"] = meta
    sweeps.save_result(r, out / f"retrieval_{llm}_{args.vision}_n{args.n}.json")
    print(json.dumps({x: r[x] for x in ("n_train", "n_test", "chance", "ridge", "relative")}, indent=1))
    return r


run.finish(summary_path=out / f"run_summary_n{args.n}.json")
