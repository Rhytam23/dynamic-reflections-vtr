import os, re, subprocess
from collections import deque
from pathlib import Path
from typing import Optional, Sequence
import logging

logger = logging.getLogger(__name__)


def run_streaming(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[dict] = None):
    """Run a command, print its output live (Colab hides child-process output otherwise),
    and on failure raise with the last lines so the real error is visible."""
    print("$ " + " ".join(map(str, cmd)), flush=True)
    env = dict(os.environ if env is None else env)
    env.setdefault("PYTHONUNBUFFERED", "1")  # child Python otherwise buffers when piped and output arrives in bursts
    proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = deque(maxlen=40)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}\n" + "".join(tail))


def venv_python(repo_path: Path) -> Path:
    return Path(repo_path) / ".venv" / "bin" / "python"


def link_to_persistent(repo_path: Path, persist_dir: Path, names=("data", "results")):
    """Keep big/slow outputs on Drive while the repo and its venv stay on the local disk.

    Drive cannot hold a venv (no symlink support), but symlinks *from* local disk *to* Drive work.
    """
    for name in names:
        target = Path(persist_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(repo_path) / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)
        print(f"Linked {link} -> {target}", flush=True)


def clone_authors_repo(repo_path: Path, persist_dir: Optional[Path] = None):
    """Clone the authors' repo (idempotent) and link data/results to Drive. No GPU or venv needed."""
    repo_url = f"https://github.com/google-deepmind/{repo_path.name}.git"
    if not (repo_path / ".git").exists():  # a directory holding only results is not a clone
        print(f"Cloning repository into {repo_path}...", flush=True)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        if repo_path.exists() and not any(repo_path.iterdir()):
            repo_path.rmdir()
        run_streaming(["git", "clone", "--depth", "1", repo_url, str(repo_path)])
    if persist_dir is not None:
        link_to_persistent(repo_path, persist_dir)


def setup_environment(repo_path: Path, persist_dir: Optional[Path] = None):
    """Build the authors' uv environment and check it imports what extraction needs."""
    clone_authors_repo(repo_path, persist_dir)
    os.chdir(repo_path)
    run_streaming(["pip", "install", "-q", "uv"])
    pyproject_path = repo_path / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(r'"transformers>=([^"]+)"', r'"transformers>=\1,<5.0.0"', content)
        pyproject_path.write_text(content)
        logger.info("Patched pyproject.toml: transformers pinned to <5.0.0")
    print("Creating and syncing uv virtual environment (uv sync downloads PyTorch etc.: 5-15 min)...", flush=True)
    run_streaming(["uv", "venv", "--python", "3.13", "--allow-existing"])
    run_streaming(["uv", "sync"])
    py = venv_python(repo_path)
    # OpenCV replaces torchcodec for video decoding (see cv2_decoder.py); install it into THIS venv explicitly.
    run_streaming(["uv", "pip", "install", "--python", str(py), "opencv-python-headless"])
    # Fail here, in minutes, if the environment cannot import what extraction needs.
    run_streaming([str(py), "-c",
                   "import cv2, torch, transformers, timm, tyro; "
                   "print('env OK | torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
                   "'| transformers', transformers.__version__, '| cv2', cv2.__version__)"])
    print("Environment setup complete.", flush=True)


def _patch_file(path: Path, old: str, new: str, marker: str, strict: bool = True) -> str:
    """Replace `old` with `new` in `path`. Idempotent (skips if `marker` is present); if the
    target text is missing, raise (strict) instead of silently patching nothing."""
    text = path.read_text()
    if marker in text:
        return "already patched"
    if old not in text:
        if strict:
            raise RuntimeError(f"Patch target not found in {path}: {old!r} (upstream code changed?)")
        return "target not found (skipped)"
    path.write_text(text.replace(old, new))
    return "patched"


PVD_DOWNLOAD_SCRIPT = """import argparse, json
from pathlib import Path
from datasets import load_dataset, Video
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_json_path', '-l', type=Path, default=None)
    parser.add_argument('--output_dir', '-o', type=Path, default='data/pvd')
    args = parser.parse_args()
    video_path = Path(args.output_dir)
    video_path.mkdir(exist_ok=True, parents=True)
    if args.local_json_path is None:
        raise SystemExit('pass --local_json_path')
    with open(args.local_json_path) as f:
        video_ids = {str(json.loads(line)['video_id']) for line in f}
    have = {p.stem for p in video_path.glob('*.bin') if p.stat().st_size > 0}
    video_ids_to_find = video_ids - have
    print(f'{len(video_ids)} videos wanted, {len(have & video_ids)} already on disk, {len(video_ids_to_find)} to download')
    if not video_ids_to_find:
        return
    ds = load_dataset('facebook/PE-Video', split='test', streaming=True)
    ds = ds.cast_column('mp4', Video(decode=False))
    with tqdm(total=len(video_ids_to_find), desc='Downloading videos') as pbar:
        for row in ds:
            if not video_ids_to_find: break
            key = row['__key__']
            if key in video_ids_to_find:
                tmp = video_path / f'{key}.bin.part'
                with open(tmp, 'wb') as f:
                    f.write(row['mp4']['bytes'] if row['mp4'].get('bytes') else open(row['mp4']['path'], 'rb').read())
                tmp.rename(video_path / f'{key}.bin')
                pbar.update(1)
                video_ids_to_find.remove(key)
    if video_ids_to_find:
        raise SystemExit(f'{len(video_ids_to_find)} videos were not found in the dataset, e.g. {sorted(video_ids_to_find)[:3]}')
if __name__ == '__main__': main()
"""

PVD_IMPORT_OLD = "from torchcodec.decoders import VideoDecoder\n"
PVD_IMPORT_NEW = (
    "try:  # OpenCV decoder first: torchcodec's CUDA build often cannot load on Colab\n"
    "  import cv2  # noqa: F401\n"
    "  from vprh.dataloaders._cv2_decoder import VideoDecoder\n"
    "except ImportError:\n"
    "  from torchcodec.decoders import VideoDecoder\n"
)


def patch_scripts(repo_path: Path):
    """Apply our fixes to the authors' repo. Idempotent; raises if a patch target has disappeared."""
    repo_path = Path(repo_path)
    results = {}

    # 1. resumable download script that works with current `datasets`
    dl = repo_path / "src" / "vprh" / "misc" / "download_pvd.py"
    dl.write_text(PVD_DOWNLOAD_SCRIPT)
    results["download_pvd.py"] = "replaced"

    # 2. timm config fix (only matters for some encoders; tolerate upstream fixing it)
    results["registry/video.py"] = _patch_file(
        repo_path / "src" / "vprh" / "registry" / "video.py",
        "resolve_data_config(self.model.pretrained_cfg, model=self.model)",
        "resolve_data_config(getattr(self.model, 'pretrained_cfg', None) or getattr(self.model, 'default_cfg', None) or {}, model=self.model)",
        marker="getattr(self.model, 'pretrained_cfg', None)", strict=False)

    # 3. OpenCV decoder instead of torchcodec (imported even for text-only extraction)
    shim_src = Path(__file__).with_name("cv2_decoder.py").read_text()
    (repo_path / "src" / "vprh" / "dataloaders" / "_cv2_decoder.py").write_text(shim_src)
    results["dataloaders/pvd.py"] = _patch_file(
        repo_path / "src" / "vprh" / "dataloaders" / "pvd.py",
        PVD_IMPORT_OLD, PVD_IMPORT_NEW, marker="_cv2_decoder")

    # 4. 10 DataLoader workers can exhaust a 12 GB Colab machine
    results["extract_features.py"] = _patch_file(
        repo_path / "src" / "vprh" / "extract_features.py",
        "num_workers=10,", "num_workers=min(4, os.cpu_count() or 2),", marker="os.cpu_count()")

    for name, status in results.items():
        print(f"patch {name}: {status}", flush=True)
    return results
