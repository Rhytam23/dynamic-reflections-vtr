from pathlib import Path
from dataclasses import dataclass

@dataclass
class PVDConfig:
    drive_base_dir: Path = Path("/content/drive/MyDrive/GenAI_Project")
    repo_name: str = "platonic_rep_video"
    dataset_output_dir: Path = Path("data/pvd")
    dataset_json_path: Path = Path("assets/pe_video_dataset_1k_rephrased.jsonl")
    vision_model_name: str = "dinov2"
    llm_model_name: str = "google/gemma-2-2b-it"
    
    def __post_init__(self):
        self.repo_full_path = self.drive_base_dir / self.repo_name
        self.repo_full_path.mkdir(parents=True, exist_ok=True)
