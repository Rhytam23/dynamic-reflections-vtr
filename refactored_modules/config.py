from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PVDConfig:
    drive_base_dir: Path = Path("/content/drive/MyDrive/GenAI_Project")
    repo_name: str = "platonic_rep_video"
    dataset_output_dir: Path = Path("data/pvd")
    dataset_json_path: Path = Path("assets/pe_video_dataset_1k_rephrased.jsonl")
    feature_dir: Path = Path("results/sample")  # features land in <feature_dir>/pvd/
    results_dir: Path = Path("results/ours")  # our JSON results and figures
    dataset: str = "pvd"
    llm_name: str = "gemma2-2b-it"  # 9B does not fit a free Colab T4
    llm_pool: str = "avg"
    vision_name: str = "dinov2_large_video"
    vision_pool: str = "cls"
    k: int = 10
    # Sweep grids. VideoMAEv2's native clip is 16 frames, so use multiples of 16 for it.
    frame_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    caption_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 10])
    video_models: List[str] = field(default_factory=lambda: ["dinov2_large_video", "videomaev2_base"])
    # The authors' extractor falls back to a model's first supported pooling and names the file
    # after it; videomaev2 only supports 'avg' (see registry/video.py pool_types).
    pool_overrides: Dict[str, str] = field(default_factory=lambda: {"videomaev2_base": "avg", "videomaev2_large": "avg"})

    def pool_for(self, video_model: str) -> str:
        return self.pool_overrides.get(video_model, self.vision_pool)

    work_dir: Path = Path("/content")  # local disk: the authors' repo and its venv live here (Drive cannot hold a venv)

    def __post_init__(self):
        self.repo_full_path = self.work_dir / self.repo_name
        # videos and features live on Drive (symlinked into the repo) so they survive a Colab disconnect
        self.persist_dir = self.drive_base_dir / "persist"
