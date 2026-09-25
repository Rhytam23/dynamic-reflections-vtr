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
    proc = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = deque(maxlen=40)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}\n" + "".join(tail))


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


def setup_environment(repo_path: Path, persist_dir: Optional[Path] = None):
    repo_url = f"https://github.com/google-deepmind/{repo_path.name}.git"
    if not repo_path.exists():
        print(f"Cloning repository into {repo_path}...", flush=True)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        run_streaming(["git", "clone", repo_url, str(repo_path)])
    os.chdir(repo_path)
    if persist_dir is not None:
        link_to_persistent(repo_path, persist_dir)
    run_streaming(["pip", "install", "-q", "uv"])
    run_streaming(["apt-get", "update", "-qq"])
    run_streaming(["apt-get", "install", "-y", "-qq", "ffmpeg"])
    pyproject_path = repo_path / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(r'"transformers>=([^"]+)"', r'"transformers>=\1,<5.0.0"', content)
        pyproject_path.write_text(content)
        logger.info("Patched pyproject.toml: transformers pinned to <5.0.0")
    print("Creating and syncing uv virtual environment (uv sync downloads PyTorch etc.: 5-15 min)...", flush=True)
    run_streaming(["uv", "venv", "--python", "3.13", "--allow-existing"])
    run_streaming(["uv", "sync"])
    print("Environment setup complete.", flush=True)

def patch_scripts(repo_path: Path):
    download_script_path = repo_path / "src" / "vprh" / "misc" / "download_pvd.py"
    if download_script_path.exists():
        fixed_script = """import argparse, json
from pathlib import Path
from datasets import load_dataset, Video
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_json_path', '-l', type=Path, default=None)
    parser.add_argument('--output_dir', '-o', type=Path, default='data/pvd')
    args = parser.parse_args()
    ds = load_dataset('facebook/PE-Video', split='test', streaming=True)
    ds = ds.cast_column('mp4', Video(decode=False))
    video_path = Path(args.output_dir)
    video_path.mkdir(exist_ok=True, parents=True)
    if args.local_json_path is not None:
        print(f'Downloading videos from {args.local_json_path}')
        with open(args.local_json_path) as f:
            video_ids = {str(json.loads(line)['video_id']) for line in f}
        video_ids_to_find = video_ids.copy()
        with tqdm(total=len(video_ids), desc='Downloading videos') as pbar:
            for row in ds:
                if not video_ids_to_find: break
                key = row['__key__']
                if key in video_ids_to_find:
                    with open(video_path / f'{key}.bin', 'wb') as f:
                        f.write(row['mp4']['bytes'] if row['mp4'].get('bytes') else open(row['mp4']['path'], 'rb').read())
                    pbar.update(1)
                    video_ids_to_find.remove(key)
if __name__ == '__main__': main()
"""
        download_script_path.write_text(fixed_script)
        logger.info("Patched download_pvd.py")
    
    video_py_path = repo_path / "src" / "vprh" / "registry" / "video.py"
    if video_py_path.exists():
        content = video_py_path.read_text()
        content = content.replace(
            "resolve_data_config(self.model.pretrained_cfg, model=self.model)",
            "resolve_data_config(getattr(self.model, 'pretrained_cfg', None) or getattr(self.model, 'default_cfg', None) or {}, model=self.model)"
        )
        video_py_path.write_text(content)
        logger.info("Patched video.py")
