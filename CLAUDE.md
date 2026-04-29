# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** `uv` · **Python:** 3.12 (locked via `.python-version`)

```bash
uv sync                                         # Install all dependencies
uv run python main.py                           # Run inference pipeline
uv run python utils/train_road_segmentation.py  # Run training
uv run python utils/plot_frameId_and_pitch.py   # Plot frame metadata
uv run python carla_module/realtime_test.py     # CARLA real-time pitch estimation
python scripts/setup_elsed.py                   # One-time: clone & patch pyelsed (uses stdlib only)
```

> Always use `uv run python` (not bare `python`) to run scripts — this ensures the `.venv` managed by uv is used. `scripts/setup_elsed.py` is the only exception because it runs before the venv exists and only uses the standard library.

No lint, format, or test commands are configured.

### First-time setup (new machine)

#### Windows (PowerShell)

```powershell
# 1. Copy env template and fill in machine-specific paths FIRST
#    Required before uv sync — the build needs OPENCV_DIR at compile time
Copy-Item .env.example .env
#    Edit .env and set:
#      CARLA_WHL_PATH  — path to carla-*.whl
#      OPENCV_BIN_PATH — e.g. C:/path/to/opencv/build/x64/vc16/bin
#      OPENCV_DIR      — e.g. C:/path/to/opencv/build  (contains OpenCVConfig.cmake)

# 2. Clone and patch pyelsed source (Python 3.12 + MSVC compatibility)
python scripts/setup_elsed.py

# 3. Build and install all dependencies
#    uv sync must have OPENCV_DIR set so CMake can find OpenCV headers
$env:OpenCV_DIR = (Select-String -Path .env -Pattern '^OPENCV_DIR=(.+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value })
uv sync

# 4. Install CARLA wheel (not on PyPI, must be installed separately)
$env:CARLA_WHL_PATH = (Select-String -Path .env -Pattern '^CARLA_WHL_PATH=(.+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value })
uv pip install $env:CARLA_WHL_PATH
```

#### Linux / macOS

```bash
# 1. Install OpenCV C++ dev package (needed to build pyelsed)
#    Ubuntu/Debian: sudo apt install libopencv-dev
#    macOS:         brew install opencv

# 2. Copy env template (only CARLA_WHL_PATH is required on Linux)
cp .env.example .env

# 3. Clone and patch pyelsed source
python scripts/setup_elsed.py

# 4. Build and install all dependencies
uv sync

# 5. Install CARLA wheel
uv pip install $CARLA_WHL_PATH
```

#### Why the order matters

- `.env` must be copied **before** `uv sync`: the pyelsed build reads `OPENCV_DIR` from the environment to locate OpenCV headers.
- `setup_elsed.py` must run **before** `uv sync`: it clones and patches the `elsed_src/` directory that uv builds as a local package.
- CARLA wheel is installed **after** `uv sync`: it is not on PyPI and must be added manually.

## Architecture

This is a monocular 3D road and lane perception system. Given a single camera image, it:
1. Segments the drivable road surface (deep learning)
2. Detects and classifies lane markings within that surface (classical CV)
3. Fits piecewise-linear lane curves and estimates camera pitch from lane geometry

### Inference pipeline (`main.py` → `utils/inference_road_lane_segmentation.py`)

Five sequential stages, all configured via `config/inference_road_lane_segmentation.yaml`:

**Stage 1 — Road segmentation** (`libs/inference/road_segmentation.py`)
- Loads a PIDNet-L checkpoint pretrained on Cityscapes (19 classes)
- Runs forward pass → interpolate to input size → argmax → road mask (class 0)
- Applies mask to the input image with bitwise AND

**Stage 2 — Lane detection** (`libs/inference/lane_segmentation.py`)
- Runs ELSED (line segment detector from `pyelsed`) on the masked road image
- Filters detected segments by slope magnitude and minimum length to remove noise

**Stage 3 — Lane classification** (`libs/inference/lane_segmentation.py`)
- `split_left_right_lines`: uses slope sign + horizontal position + `x_at_bottom` projection to assign segments to left/right lanes
- A `lane_band_tolerance` pixel window around the innermost segment keeps only the ego-lane markings
- Note: K-means clustering (on slope + mid-y) is commented out — it was replaced by this heuristic because K-means couldn't handle missing-lane or curved-slope scenarios

**Stage 4 — Lane fitting** (`libs/inference/lane_fitting.py`)
- Densifies each cluster into point clouds (`collect_points_from_segments`)
- `piecewise_linear_fit`: divides the y-range into `num_bands` horizontal bands and fits `x = f(y)` per band using `np.polyfit`
- RANSAC was tried but discarded (commented out) — far bands have too few points for a fixed residual threshold
- `compute_lane_widths`: samples the overlapping y-range of left/right fits to produce `(y, pixel_width)` pairs

**Stage 5 — Pitch estimation** (`libs/inference/pitch_estimation.py`)
- Converts pixel lane widths to metric depth via pinhole model: `depth = f_x * w_real / pixel_width`
- Lifts road points to 3D Y coordinate: `Y_3d = -depth * (y_pixel - center_y) / f_y`
- Fits depth→Y with Theil-Sen (robust to outliers) and takes `arctan(slope)` as pitch
- IQR filter applied to pixel widths before depth conversion to suppress bad lane-fitting samples

### Training pipeline (`utils/train_road_segmentation.py`)

Configured via `config/train_road_segmentation.yaml`.
- Dataset: Cityscapes, loaded by `libs/dataset/cityscape_dataset.py`, resized to 512×1024
- `ToRoadMask()` transform converts Cityscapes label IDs to a binary road mask (class 0 = road)
- Model: DeepLabV3-ResNet101 with a single-channel output head (`libs/model/resnet101.py`)
- Loss: BCE on main output + weighted auxiliary classifier output (`aux_loss_weight=0.4`)
- Best checkpoint saved to `models/` when validation IoU improves

### Key config parameters (`config/inference_road_lane_segmentation.yaml`)

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `f_x`, `f_y` | `pitch_estimation` | Focal lengths (pixels) after resize — derived from camera intrinsics |
| `w_real` | `pitch_estimation` | Known real-world lane width in metres (3.5 m for CARLA) |
| `lane_band_tolerance` | `lane_segmentation` | Max pixel deviation at image bottom to still count as ego lane |
| `num_bands` | `lane_fitting` | Number of horizontal bands for piecewise fit |

### Module map
| Path | Role |
|------|------|
| `libs/model/resnet101.py` | `build_train_model()` / `build_inference_model()` — wraps TorchVision DeepLabV3 |
| `libs/dataset/cityscape_dataset.py` | Dataset loader + `ToRoadMask` transform |
| `libs/engine/train.py` / `validate.py` | Per-epoch training and IoU validation loops |
| `libs/inference/road_segmentation.py` | `load_pidnet()` / `predict_road()` / `apply_road_mask()` — PIDNet inference |
| `libs/inference/lane_segmentation.py` | ELSED detection + slope/position lane classification |
| `libs/inference/lane_fitting.py` | Point collection, piecewise linear fitting, lane width computation |
| `libs/inference/pitch_estimation.py` | Pinhole-model depth + Theil-Sen pitch estimation |
| `libs/metric/iou.py` | IoU computation used during validation |
| `libs/visualization/lane_visualization.py` | Loss curve, lane overlay, and piecewise-fit plotting (file output) |
| `config/` | YAML configs for train and inference (paths, hyperparameters, device) |
| `outputs/` | Inference result images |
| `results/` | Training loss plots |
| `models/` | Saved model checkpoints |
| `pidnet_models/` | PIDNet model architecture (pidnet.py, model_utils.py) and speed variants |
| `pidnet_pretrained_model/` | PIDNet pretrained weights (see readme.txt for download instructions) |
| `utils/env_setup.py` | Shared startup helper: loads `.env` and registers OpenCV DLL path on Windows |
| `carla_module/realtime_test.py` | Real-time CARLA test: spawns vehicle, runs 5-stage pipeline, displays pitch overlay |
| `carla_module/carla_road_segmentation.py` | `predict_road_from_pil()` — PIDNet version accepting PIL Image (CARLA streaming) |
| `carla_module/carla_visualization.py` | `render_piecewise_fits_to_array()` — returns BGR ndarray for `cv2.imshow` |
| `scripts/setup_elsed.py` | One-time setup: clones pyelsed and applies Python 3.12 / Windows patches |
| `.env.example` | Template for machine-specific paths (`CARLA_WHL_PATH`, `OPENCV_BIN_PATH`, `OPENCV_DIR`) |

### CARLA camera parameters

`carla_module/realtime_test.py` uses `image_size_x=1024`, `image_size_y=512`, `fov=90°`, which gives `f_x = f_y = 512` (square pixels). The config value `f_y=455` (from a real camera) is overridden at runtime to match CARLA's pinhole model.

### pyelsed build notes (Windows + Python 3.12)

`scripts/setup_elsed.py` applies five patches to the cloned source before `uv sync` builds it:

1. `CMakeLists.txt` line 1 → `cmake_minimum_required(VERSION 3.5)`
2. `CMakeLists.txt` — injects `add_compile_definitions(_USE_MATH_DEFINES)` inside an `if(MSVC)` block (fixes `M_PI` / `M_PI_2` undefined across all `.cpp` files on MSVC)
3. `src/EdgeDrawer.h` top → `#define _USE_MATH_DEFINES` (belt-and-suspenders for older CMake flows)
4. `setup.py` — injects code to read `OpenCV_DIR` / `OPENCV_DIR` env var and pass it as `-DOpenCV_DIR` to CMake (needed because `opencv-python` on PyPI ships only Python bindings, not C++ headers or `OpenCVConfig.cmake`)
5. `pybind11` submodule → checkout `v2.13.6` (fixes Python 3.12 C-API incompatibility)

Set `OPENCV_DIR` in `.env` to the directory containing `OpenCVConfig.cmake` (e.g. `C:/path/to/opencv/build`). Not needed on macOS/Linux where `brew`/`apt` installs headers to a system path CMake finds automatically.
