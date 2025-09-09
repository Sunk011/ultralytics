# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CLI Commands

The Ultralytics CLI uses the following syntax:
```bash
yolo TASK MODE ARGS
```

Where:
- `TASK` (optional) is one of [detect, segment, classify, pose, obb]
- `MODE` (required) is one of [train, val, predict, export, track, benchmark]
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults.

### Common CLI Commands

**Training:**
```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

**Validation:**
```bash
yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
```

**Prediction:**
```bash
yolo predict model=yolo11n-seg.pt source='path/to/image.jpg' imgsz=320
```

**Export:**
```bash
yolo export model=yolo11n.pt format=onnx imgsz=640
```

**Tracking:**
```bash
yolo track model=yolo11n.pt source='0'  # webcam
yolo track model=yolo11n.pt source='path/to/video.mp4'
```

**Special Commands:**
```bash
yolo help      # Show help message
yolo checks # Run system checks
yolo version # Show version
yolo settings  # Manage settings
yolo cfg       # Show default configuration
yolo solutions # Show available solutions
```

## Testing Commands

### Running Tests

The project uses pytest for testing. Key test commands:

**Run all tests:**
```bash
python -m pytest tests/
```

**Run specific test files:**
```bash
python -m pytest tests/test_cli.py
python -m pytest tests/test_engine.py
```

**Run tests with slow tests included:**
```bash
python -m pytest tests/ --slow
```

**Run tests for specific tasks:**
```bash
# Run CLI tests
python -m pytest tests/test_cli.py -v

# Run engine tests  
python -m pytest tests/test_engine.py -v

# Run Python API tests
python -m pytest tests/test_python.py -v
```

**Test configuration from pyproject.toml:**
- `--doctest-modules`: Run doctests in all modules
- `--durations=30`: Show slowest 30 test durations
- `--color=yes`: Enable colored output
- `slow: skip slow tests unless --slow is set`: Skip slow tests by default

## Build and Development Commands

### Installation

**Install from source in development mode:**
```bash
pip install -e .
```

**Install with optional dependencies:**
```bash
# Install with all extras
pip install -e .[dev]

# Install specific extras (examples)
pip install -e .[dev,logging]
```

### Building Documentation

**Build docs locally:**
```bash
cd docs && python build_docs.py
```

**Serve docs with live reload:**
```bash
cd docs && mkdocs serve
```

### Code Formatting and Linting

**Run ruff linter:**
```bash
ruff check ultralytics/
```

**Auto-format with ruff:**
```bash
ruff format ultralytics/
```

**Run codespell for spelling:**
```bash
codespell ultralytics/
```

## Codebase Architecture Overview

### Core Structure

**Main Modules:**
- `ultralytics/`: Core YOLO modules
  - `cfg/`: Configuration and CLI entrypoint
  - `data/`: Data loading, augmentation, and processing
  - `engine/`: Core model lifecycle (Model, Trainer, Validator, Predictor, Exporter)
  - `models/`: YOLO model variants (YOLO, YOLOE, YOLOWorld, SAM, FastSAM, RTDETR, NAS)
  - `nn/`: Neural network architecture and modules
  - `solutions/`: Ready-to-use computer vision solutions
  - `trackers/`: Object tracking implementations
  - `utils/`: Utility functions and helpers

**Configuration:**
- `ultralytics/cfg/`: Default configurations, CLI parsing, and settings
- `ultralytics.egg-info/entry_points.txt`: CLI entry points (`yolo` and `ultralytics` commands)

**Testing:**
- `tests/`: Unit and integration tests organized by functionality
- `tests/conftest.py`: Pytest configuration and hooks

**Documentation:**
- `docs/`: MkDocs documentation with comprehensive guides
- `docs/en/`: English documentation with usage examples

**Examples and Usage:**
- `examples/`: Example implementations and integrations
- `usage/`: Usage examples and training results
- `code/`: Additional implementation examples

### Key Entry Points

**CLI:** `ultralytics.cfg.entrypoint` (accessible via `yolo` and `ultralytics` commands)
**Python API:** `ultralytics.YOLO()` - main Python interface
**Configuration:** `ultralytics.cfg.get_cfg()` - configuration management

## Common Workflows

### Model Training Workflow

1. **Prepare data:**
 ```bash
   # Data should be in YOLO format with proper YAML config
   yolo train data=your_dataset.yaml model=yolo11n.pt epochs=100
   ```

2. **Monitor training:** Check TensorBoard logs in `runs/train/exp*/`

3. **Validate model:**
   ```bash
   yolo val model=runs/train/exp/weights/best.pt data=your_dataset.yaml
   ```

4. **Export for deployment:**
   ```bash
   yolo export model=runs/train/exp/weights/best.pt format=onnx
   ```

5. **Test exported model:**
   ```bash
   yolo predict model=runs/train/exp/weights/best.onnx source=test_images/
   ```

### Development Workflow

1. **Set up development environment:**
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics && pip install -e .[dev]
   ```

2. **Make code changes:** Edit files and test locally

3. **Run tests:**
   ```bash
   python -m pytest tests/test_engine.py -v # Run specific tests
   python -m pytest tests/ --slow          # Run all tests including slow ones
   ```

4. **Check formatting and linting:**
   ```bash
   ruff check ultralytics/ && ruff format ultralytics/
   ```

5. **Build and test package:**
   ```bash
   python -m build
   ```

### Multi-Task Development

The codebase supports multiple tasks:
- **Detection:** `task=detect` (default)
- **Segmentation:** `task=segment` 
- **Classification:** `task=classify`
- **Pose Estimation:** `task=pose`
- **Oriented Bounding Boxes:** `task=obb`

Example for different tasks:
```bash
yolo train detect data=coco8.yaml model=yolo11n.pt
yolo train segment data=coco8-seg.yaml model=yolo11n-seg.pt 
yolo train classify data=imagenet10 model=yolo11n-cls.pt
```

### Solutions Development

The codebase includes ready-to-use solutions:
- `ultralytics.solutions.*`: Various computer vision applications
- Use `yolo solutions` to see available solutions
- Each solution can be used directly or customized

## Project Structure Highlights

- **Modular Design:** Clean separation of concerns across modules
- **Multiple APIs:** CLI, Python, and direct module imports
- **Extensible Architecture:** Easy to add new models, tasks, and solutions
- **Comprehensive Testing:** Extensive test coverage with pytest
- **Production Ready:** Export to ONNX, TensorRT, TensorFlow, etc.
- **Documentation:** Complete documentation with examples and guides

## Version Information

- Current version: `ultralytics.__version__` (check via `yolo version`)
- License: AGPL-3.0
- Python requirements: >=3.8

## Quick Start

1. **Install:** `pip install ultralytics`
2. **Try CLI:** `yolo predict model=yolo11n.pt source="https://ultralytics.com/images/zidane.jpg"`
3. **Try Python:** 
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n.pt')
   results = model.predict('https://ultralytics.com/images/zidane.jpg')
   ```
4. **Train:** `yolo train data=coco8.yaml model=yolo11n.pt epochs=10`

For more information, see the [Ultralytics Docs](https://docs.ultralytics.com).