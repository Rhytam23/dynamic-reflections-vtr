import glob, os, re, subprocess
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


def venv_python(repo_path: Path) -> Path:
    return Path(repo_path) / ".venv" / "bin" / "python"


def find_library_dirs(repo_path: Path, name: str = "libnppicc.so") -> list:
    """Directories that actually contain `name*` (venv first, then system CUDA locations).

    Searches the file system, so call it once during setup, not on every command.
    """
    dirs = []
    for root in (Path(repo_path) / ".venv", Path("/usr/local"), Path("/usr/lib"), Path("/opt")):
        if root.exists():
            for hit in root.rglob(name + "*"):
                d = str(hit.parent)
                if d not in dirs:
                    dirs.append(d)
    return dirs


def cuda_library_env(repo_path: Path, base_env: Optional[dict] = None, extra_dirs: Sequence[str] = ()) -> dict:
    """Environment with the venv's nvidia/*/lib dirs (and system CUDA) on LD_LIBRARY_PATH.

    torchcodec's CUDA build needs libnppicc.so.12, which is not on the loader path inside the
    uv venv on Colab ("Could not load libtorchcodec").
    """
    env = dict(os.environ if base_env is None else base_env)
    dirs = list(extra_dirs)
    dirs += sorted(glob.glob(str(Path(repo_path) / ".venv" / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")))
    dirs += [d for d in ("/usr/local/cuda/lib64", "/usr/lib64-nvidia") if os.path.isdir(d)]
    if env.get("LD_LIBRARY_PATH"):
        dirs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(dirs))
    return env


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
    # torchcodec's CUDA build needs libnppicc.so.12. Install into THIS venv explicitly (a plain
    # `uv pip install` did not land in it on Colab), find where the library really is, expose it
    # to every later command in this session, and fail here (not after a long extraction) if
    # torchcodec still cannot load.
    py = venv_python(repo_path)
    run_streaming(["uv", "pip", "install", "--python", str(py), "nvidia-npp-cu12"])
    lib_dirs = find_library_dirs(repo_path)
    print(f"libnppicc found in: {lib_dirs or 'NOWHERE'}", flush=True)
    os.environ.update(cuda_library_env(repo_path, extra_dirs=lib_dirs))
    run_streaming([str(py), "-c", "import torchcodec; print('torchcodec OK', torchcodec.__version__)"], env=dict(os.environ))
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
