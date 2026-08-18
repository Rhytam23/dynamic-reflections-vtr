import os, re, subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment(repo_path: Path):
    repo_url = f"https://github.com/google-deepmind/{repo_path.name}.git"
    if not repo_path.exists():
        logger.info(f"Cloning repository into {repo_path}...")
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
    os.chdir(repo_path)
    subprocess.run(["pip", "install", "-q", "uv"], check=True)
    subprocess.run(["apt-get", "update", "-qq", "&&", "apt-get", "install", "-y", "-qq", "ffmpeg"], shell=True, check=True)
    pyproject_path = repo_path / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(r'"transformers>=([^"]+)"', r'"transformers>=4.0.0,<5.0.0"', content)
        pyproject_path.write_text(content)
        logger.info("Patched pyproject.toml: transformers pinned to <5.0.0")
    logger.info("Creating and syncing uv virtual environment...")
    subprocess.run(["uv", "venv", "--python", "3.10"], check=True)
    subprocess.run(["uv", "sync"], check=True)
    logger.info("Environment setup complete.")

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
