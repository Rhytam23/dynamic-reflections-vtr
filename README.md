# Dynamic Reflections Notebook: Code Analysis & Refactoring Roadmap

## Executive Summary
Your notebook (`colab_run_pfinal.ipynb`) implements the PVD sample experiment pipeline for the Dynamic Reflections paper. It has **24 cells** (16 code, 8 markdown) and spans ~1,300 lines. The code is functional but has opportunities for refactoring to improve maintainability, reusability, and performance.

---

## 1. Current Architecture Overview

### Workflow Pipeline
```
Setup & Auth
    ↓
Environment Setup (Git clone, install deps, patch files)
    ↓
Download Dataset (PVD 1K videos)
    ↓
Feature Extraction (Gemma2 LLM + DINOv2 vision)
    ↓
Alignment Measurement (Mutual KNN)
    ↓
Inference & Visualization
```

### Key Sections
| Cell | Type | Purpose |
|------|------|---------|
| 1-2 | Setup | Google Drive mount, HF login |
| 3-6 | Environment | Clone repo, patch dependencies, install packages |
| 7-8 | Data | Download PVD dataset, display sample video |
| 9-11 | Feature Extraction | Extract LLM & vision features, explain alignment metric |
| 12-13 | Results | Display alignment plots |
| 14-16 | Analysis & Inference | Load embeddings, visualization, model inference |

---

## 2. Code Quality Issues & Anti-Patterns

### 🔴 High Priority Issues

#### 2.1 **Magic Paths & Hard-Coded Values**
**Location**: Throughout (cells 4, 9, 12, 15, 16)

```python
# Problems:
'/content/drive/MyDrive/GenAI_Project'  # Hard-coded path
'data/pvd'  # Relative path, fragile
'results/alignment/sample/pvd/*.png'  # Mixed separators
'assets/pe_video_dataset_1k_rephrased.jsonl'  # No existence check
```

**Impact**: Code breaks on different machines/directory structures

**Refactoring**: Create centralized `Config` class
```python
class PVDConfig:
    DRIVE_BASE = Path('/content/drive/MyDrive/GenAI_Project')
    DATA_DIR = Path('data/pvd')
    RESULTS_DIR = Path('results/alignment/sample/pvd')
    ASSETS_DIR = Path('assets')
    DATASET_FILE = ASSETS_DIR / 'pe_video_dataset_1k_rephrased.jsonl'
```

---

#### 2.2 **Mixed Concerns: Shell Commands + Python**
**Location**: Cells 3, 4, 7

```python
# Current:
!git clone https://github.com/google-deepmind/platonic_rep_video.git
%cd platonic_rep_video
!pip install uv
!apt-get update -qq && apt-get install -y ffmpeg -qq
!HF_TOKEN={hf_token} uv run scripts/main_extract.py pvd_sample --llm_only
```

**Problems**:
- Hard to test or mock
- Shell syntax is fragile and platform-dependent
- Error handling is poor
- Mixing Python and bash logic

**Refactoring**: Wrap in utility functions
```python
def setup_environment():
    """Clone repo, install dependencies, patch files."""
    subprocess.run(['git', 'clone', ...], check=True)
    subprocess.run(['pip', 'install', 'uv'], check=True)
    # ... etc

def run_feature_extraction(hf_token: str, extract_type: str):
    """Run uv command with proper error handling."""
    env = os.environ.copy()
    env['HF_TOKEN'] = hf_token
    subprocess.run(['uv', 'run', 'scripts/main_extract.py', ...], 
                   env=env, check=True)
```

---

#### 2.3 **Monolithic Code Blocks**
**Location**: Cell 15 (Analysis & Inference), Cell 16 (Inference)

These cells are **massive** (100+ lines each) and do multiple things:
- Load embeddings
- Extract video frames
- Compute alignments
- Visualize results

**Problem**: Hard to test, debug, or reuse individual components

**Refactoring**: Break into focused functions
```python
def load_embeddings(gemma_layer: int, dino_layer: int) -> Tuple[Tensor, Tensor]:
    """Load pre-extracted embeddings for specific layers."""
    
def extract_video_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """Extract and decode frames from video file."""
    
def compute_alignment_score(llm_feats, vision_feats, k: int = 10) -> float:
    """Compute mutual KNN alignment metric."""
    
def visualize_alignment(video_id: str, caption: str, frames: List, scores: Dict):
    """Create multi-panel visualization."""
```

---

#### 2.4 **No Error Handling**
**Location**: Throughout

```python
# Current (problematic):
cap = cv2.VideoCapture(video_file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Issues: What if video_file doesn't exist? What if cv2 fails?
```

**Refactoring**: Add try-except and validation
```python
def extract_video_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """Extract evenly-spaced frames from video."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # ... rest of logic
    except Exception as e:
        logger.error(f"Frame extraction failed for {video_path}: {e}")
        raise
    finally:
        cap.release()
```

---

#### 2.5 **Global State & Dependencies**
**Location**: Cell 9 (HF token), Cell 16 (Model loading)

```python
# Implicit global state:
hf_token = get_token()  # Where is this used? Multiple places?
language_model = llm_registry.create_llm(...)  # Loaded once, used implicitly
video_model = video_registry.VIDEO_MODEL_REGISTRY[...]  # Same issue
```

**Problem**: Unclear dependencies, hard to mock for testing, GPU memory not managed

**Refactoring**: Use dependency injection
```python
class ModelManager:
    def __init__(self, device: str = 'cuda', quantize: bool = False):
        self.device = device
        self.quantize = quantize
        self._llm = None
        self._video = None
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._load_llm()
        return self._llm
    
    def cleanup(self):
        """Free GPU memory."""
        if self._llm is not None:
            del self._llm
        torch.cuda.empty_cache()
```

---

#### 2.6 **Inconsistent File Handling**
**Location**: Cells 5, 8, 15, 16

```python
# Mixing pathlib and os.path:
os.path.join(drive_base_dir, repo_name)  # os.path
Path(video_file).replace('.bin', '.mp4')  # pathlib
open(jsonl_path, 'r') as f:  # string path
glob.glob(f"data/pvd/{video_id}.bin")  # string path with f-string
```

**Refactoring**: Use pathlib consistently
```python
from pathlib import Path

video_file = Path('data/pvd') / f'{video_id}.bin'
mp4_file = video_file.with_suffix('.mp4')
results_dir = Path('results/alignment/sample/pvd')
for png in results_dir.glob('*.png'):
    print(png)
```

---

### 🟡 Medium Priority Issues

#### 2.7 **Duplicated Logic**
- Cell 12 & 13: Both try to locate alignment plots
- Cell 9 & 10: Feature extraction separated into two cells for LLM and vision
- Multiple places load the same datasets

**Solution**: Extract into helper functions, memoize expensive operations

---

#### 2.8 **No Logging**
```python
# Current (hard to debug):
print("Mounting drive...")
print("Error: token not found")

# Better:
import logging
logger = logging.getLogger(__name__)
logger.info("Mounting Google Drive")
logger.error("HF token not found: %s", e)
```

---

#### 2.9 **Missing Type Hints**
**Location**: All function-like code blocks

```python
# Current:
def extract_video_feature(video_path, num_frames):
    # What type is video_path? str? Path?
    # What does this return?

# Better:
def extract_video_feature(video_path: Union[str, Path], num_frames: int = 5) -> torch.Tensor:
    """Extract DINOv2 features for num_frames from video."""
```

---

#### 2.10 **Notebook-Specific Anti-Patterns**
- Cell ordering matters, but code doesn't validate prerequisites
- No way to selectively run pipeline stages
- Output cells have execution counts, making it unclear what's been run

---

## 3. Data Flow Issues

### Current Data Flow (Implicit)
```
Load embeddings (cell 15)
    ↓ (manual tensor loading)
Extract video frames (cell 15)
    ↓ (hardcoded first_sample logic)
Compute alignment (implicit in plot generation)
    ↓
Visualize (cell 15 & 16)
```

**Problems**:
- Dependencies are implicit (based on cell execution order)
- No validation that intermediate outputs exist
- Hard to run just one stage in isolation

**Refactoring Approach**:
```python
class PVDPipeline:
    """Orchestrate the full analysis pipeline."""
    
    def __init__(self, config: PVDConfig, models: ModelManager):
        self.config = config
        self.models = models
    
    def run_feature_extraction(self) -> Dict[str, Path]:
        """Extract LLM & vision features, return paths."""
        
    def load_embeddings(self, gemma_layer: int, dino_layer: int) -> Tuple[Tensor, Tensor]:
        """Load pre-computed embeddings."""
        
    def run_inference(self, video_id: str) -> Dict:
        """End-to-end inference: extract features, compute alignment, visualize."""
        
    def run_full_pipeline(self) -> Dict:
        """Execute all stages in sequence."""
```

---

## 4. Refactoring Priorities & Roadmap

### Phase 1: Foundation (Do First)
**Effort**: 2-3 hours | **Impact**: High

1. **Extract config to standalone class** (`config.py`)
   - Centralize all paths, hyperparameters, model names
   - Enables parameterization without editing notebook

2. **Create `utils.py` with utility functions**
   - `setup_environment()` - wrap shell commands
   - `extract_video_frames()` - reusable frame extraction
   - `load_embeddings()` - centralized loading logic
   - Add logging

3. **Add type hints & docstrings**
   - Every function: input types, return types, purpose
   - Helps catch bugs early

4. **Switch to pathlib consistently**
   - Replace all `os.path.join()` with `Path()`
   - Use `.glob()`, `.exists()`, `.mkdir(parents=True)`

---

### Phase 2: Modularity (Do Second)
**Effort**: 3-4 hours | **Impact**: Medium

5. **Extract analysis functions into `analysis.py`**
   - `load_and_prepare_embeddings()`
   - `compute_alignment_metrics()`
   - `visualize_alignment_results()`

6. **Create `ModelManager` class**
   - Centralize model loading/unloading
   - Handle GPU memory management
   - Enable model switching

7. **Refactor pipeline orchestration**
   - Create `PVDPipeline` class
   - Support selective stage execution
   - Add checkpointing (skip re-running expensive stages)

---

### Phase 3: Testing & Documentation (Do Third)
**Effort**: 2-3 hours | **Impact**: Medium

8. **Add unit tests**
   - Test `extract_video_frames()` with mock video
   - Test `compute_alignment_metrics()` with synthetic embeddings
   - Test config loading/validation

9. **Create README with architecture docs**
   - Component overview
   - How to extend with new models/datasets
   - Troubleshooting guide

10. **Convert notebook to script + notebook**
    - Main script: `run_pvd_experiment.py` (importable, testable)
    - Notebook: Thin wrapper calling script functions (visualization only)

---

## 5. Code Organization Proposal

### Recommended Structure
```
dynamic-reflections-vtr/
├── colab_run_pfinal.ipynb  (↓ thin wrapper, calls functions from below)
├── run_pvd_experiment.py   (↑ main entry point, orchestrates pipeline)
│
├── vprh/                   (existing)
├── assets/                 (existing)
├── results/                (existing)
│
└── refactored_modules/     (new)
    ├── __init__.py
    ├── config.py           # PVDConfig class + validation
    ├── models.py           # ModelManager class
    ├── pipeline.py         # PVDPipeline orchestrator
    ├── analysis.py         # Analysis functions
    ├── utils.py            # Utility functions (setup, frame extraction, etc.)
    ├── logging_setup.py    # Logger configuration
    └── tests/
        ├── test_config.py
        ├── test_utils.py
        └── test_pipeline.py
```

---

## 6. Example Refactoring: Cell 15 → Modular Functions

### Before (Current Monolithic Cell)
```python
# Cell 15: 100+ lines mixing loading, extraction, computation, visualization

import torch
import json
import matplotlib.pyplot as plt
# ... 10 more imports

jsonl_path = "assets/pe_video_dataset_1k_rephrased.jsonl"
with open(jsonl_path, 'r') as f:
    first_sample = json.loads(f.readline())
video_id = first_sample['video_id']
caption = first_sample['rephrased_captions'][0]

# Extract frames
video_file = glob.glob(f"data/pvd/{video_id}.bin")[0]
cap = cv2.VideoCapture(video_file)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Load embeddings
gemma_files = glob.glob("/content/platonic_rep_video/platonic_rep_video/results/sample/pvd/gemma2-9b-it_pool-avg.pt")
dino_files = glob.glob("/content/platonic_rep_video/platonic_rep_video/results/sample/pvd/dinov2_large_video*.pt")
reference_llm_feats = torch.load(gemma_files[0], map_location='cuda', weights_only=False)['feats'][:, 38, :].float()
reference_vision_feats = torch.load(dino_files[0], map_location='cuda', weights_only=False)['feats'][:, 22, :].float()

# ... compute alignment, visualize
```

### After (Refactored - Modular)
```python
# refactored_modules/config.py
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PVDConfig:
    DRIVE_BASE: Path = Path('/content/drive/MyDrive/GenAI_Project')
    DATA_DIR: Path = Path('data/pvd')
    RESULTS_DIR: Path = Path('results/alignment/sample/pvd')
    ASSETS_DIR: Path = Path('assets')
    DATASET_FILE: Path = ASSETS_DIR / 'pe_video_dataset_1k_rephrased.jsonl'
    GEMMA_LAYER: int = 38
    DINO_LAYER: int = 22
    ALIGNMENT_K: int = 10

# refactored_modules/utils.py
from pathlib import Path
import json
import cv2
import torch
import logging

logger = logging.getLogger(__name__)

def load_sample_metadata(dataset_file: Path) -> Dict[str, Any]:
    """Load first video's metadata from JSONL."""
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    try:
        with open(dataset_file, 'r') as f:
            sample = json.loads(f.readline())
        return {
            'video_id': sample['video_id'],
            'caption': sample['rephrased_captions'][0]
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load metadata: {e}")
        raise

def extract_video_frames(video_path: Union[str, Path], num_frames: int = 5) -> List[np.ndarray]:
    """Extract evenly-spaced frames from video file."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    finally:
        cap.release()

def load_embeddings(results_dir: Path, gemma_layer: int, dino_layer: int) -> Tuple[Tensor, Tensor]:
    """Load pre-extracted embeddings for specific layers."""
    results_dir = Path(results_dir)
    
    gemma_file = list(results_dir.glob("gemma2-9b-it_pool-avg.pt"))
    dino_file = list(results_dir.glob("dinov2_large_video*.pt"))
    
    if not gemma_file or not dino_file:
        raise FileNotFoundError(f"Embedding files not found in {results_dir}")
    
    gemma_feats = torch.load(gemma_file[0], map_location='cuda', weights_only=False)['feats'][:, gemma_layer, :].float()
    dino_feats = torch.load(dino_file[0], map_location='cuda', weights_only=False)['feats'][:, dino_layer, :].float()
    
    # Normalize
    gemma_feats = F.normalize(gemma_feats, p=2, dim=-1)
    dino_feats = F.normalize(dino_feats, p=2, dim=-1)
    
    logger.info(f"Loaded embeddings: gemma {gemma_feats.shape}, dino {dino_feats.shape}")
    return gemma_feats, dino_feats

# refactored_modules/analysis.py
def visualize_sample_analysis(config: PVDConfig):
    """End-to-end analysis: load, extract, visualize."""
    # Load metadata
    metadata = load_sample_metadata(config.DATASET_FILE)
    video_id = metadata['video_id']
    caption = metadata['caption']
    
    # Extract frames
    video_file = config.DATA_DIR / f"{video_id}.bin"
    frames = extract_video_frames(video_file, num_frames=5)
    
    # Load embeddings
    gemma_feats, dino_feats = load_embeddings(
        config.RESULTS_DIR,
        gemma_layer=config.GEMMA_LAYER,
        dino_layer=config.DINO_LAYER
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f"Frame {i}")
    plt.tight_layout()
    plt.show()
    
    return {'video_id': video_id, 'caption': caption, 'frames': frames}

# Cell 15 in notebook (now clean & simple):
from refactored_modules.config import PVDConfig
from refactored_modules.analysis import visualize_sample_analysis

config = PVDConfig()
result = visualize_sample_analysis(config)
```

---

## 7. Quick Wins (Do First)

If you want immediate improvements without major refactoring:

### ✅ Quick Win 1: Add Config Class (15 min)
Create `config.py` with all hard-coded paths/values. Import in notebook.

### ✅ Quick Win 2: Extract Frame Extraction (20 min)
Move video frame extraction to `utils.py`, add error handling, import in notebook.

### ✅ Quick Win 3: Add Logging (10 min)
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting pipeline...")
```

### ✅ Quick Win 4: Use pathlib (20 min)
Find & replace all path operations with `Path()` objects.

### ✅ Quick Win 5: Add Type Hints (30 min)
Add type annotations to all function signatures.

---

## 8. Estimated Effort & Impact

| Refactoring | Effort | Impact | Do When |
|------------|--------|--------|---------|
| Config class | 15 min | ⭐⭐ | Now |
| Extract utilities | 45 min | ⭐⭐⭐ | Now |
| Pathlib + type hints | 30 min | ⭐⭐ | Now |
| Error handling | 1 hour | ⭐⭐⭐ | Phase 1 |
| Pipeline orchestrator | 2 hours | ⭐⭐⭐ | Phase 2 |
| Unit tests | 1.5 hours | ⭐⭐ | Phase 3 |
| **Total** | **~6 hours** | **⭐⭐⭐** | |

---

## Summary: Next Steps

1. **Review this analysis** - Identify which issues matter most for your use case
2. **Start with Phase 1** - Config, utils, type hints, pathlib
3. **Decide on script approach** - Keep as notebook? Move to `.py` file? Hybrid?
4. **Create test infrastructure** - Even simple mocking will help catch bugs early
5. **Document as you go** - Add docstrings & README explaining new structure

Would you like me to implement any specific refactoring? I can:
- Create `config.py` with your paths
- Extract `utils.py` with reusable functions
- Build a `pipeline.py` orchestrator
- Write a starter test suite

Let me know what you'd like to focus on first!
