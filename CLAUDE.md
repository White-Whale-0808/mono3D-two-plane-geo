# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** `uv` · **Python:** 3.12 (locked via `.python-version`)

```bash
uv sync                                    # Install all dependencies
python main.py                             # Run inference pipeline
python utils/train_road_segmentation.py   # Run training
python utils/plot_frameId_and_pitch.py    # Plot frame metadata
python carla_realtime_test.py             # CARLA real-time pitch estimation
python scripts/setup_elsed.py             # One-time: clone & patch pyelsed for Python 3.12
```

No lint, format, or test commands are configured.

### First-time setup (new machine)

```bash
# 1. Patch and build pyelsed locally (required for Python 3.12 on Windows)
python scripts/setup_elsed.py

# 2. Copy env template and fill in machine-specific paths
cp .env.example .env

# 3. Install CARLA wheel (path set in .env)
#    PowerShell: uv pip install $env:CARLA_WHL_PATH
#    bash:       uv pip install $CARLA_WHL_PATH

# 4. Install all other dependencies
uv sync
```

## Architecture

This is a monocular 3D road and lane perception system. Given a single camera image, it:
1. Segments the drivable road surface (deep learning)
2. Detects and classifies lane markings within that surface (classical CV)
3. Fits piecewise-linear lane curves and estimates camera pitch from lane geometry

### Inference pipeline (`main.py` → `utils/inference_road_lane_segmentation.py`)

Five sequential stages, all configured via `config/inference_road_lane_segmentation.yaml`:

**Stage 1 — Road segmentation** (`libs/inference/road_segmentation.py`)
- Loads a trained DeepLabV3-ResNet101 checkpoint
- Runs forward pass → sigmoid → threshold → binary mask
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
| `libs/inference/road_segmentation.py` | Inference-time road masking |
| `libs/inference/carla_road_segmentation.py` | `predict_road_from_pil()` — accepts PIL Image directly (used by CARLA script) |
| `libs/inference/lane_segmentation.py` | ELSED detection + slope/position lane classification |
| `libs/inference/lane_fitting.py` | Point collection, piecewise linear fitting, lane width computation |
| `libs/inference/pitch_estimation.py` | Pinhole-model depth + Theil-Sen pitch estimation |
| `libs/metric/iou.py` | IoU computation used during validation |
| `libs/visualization/lane_visualization.py` | Loss curve, lane overlay, and piecewise-fit plotting (file output) |
| `libs/visualization/carla_visualization.py` | `render_piecewise_fits_to_array()` — returns BGR ndarray for `cv2.imshow` |
| `config/` | YAML configs for train and inference (paths, hyperparameters, device) |
| `outputs/` | Inference result images |
| `results/` | Training loss plots |
| `models/` | Saved model checkpoints |
| `carla_realtime_test.py` | Real-time CARLA test: spawns vehicle, runs 5-stage pipeline, displays pitch overlay |
| `scripts/setup_elsed.py` | One-time setup: clones pyelsed and applies Python 3.12 / Windows patches |
| `.env.example` | Template for machine-specific paths (`CARLA_WHL_PATH`, `OPENCV_BIN_PATH`) |

### CARLA camera parameters

`carla_realtime_test.py` uses `image_size_x=1024`, `image_size_y=512`, `fov=90°`, which gives `f_x = f_y = 512` (square pixels). The config value `f_y=455` (from a real camera) is overridden at runtime to match CARLA's pinhole model.

### pyelsed build notes (Windows + Python 3.12)

`scripts/setup_elsed.py` applies three patches to the cloned source before `uv sync` builds it:

1. `CMakeLists.txt` line 1 → `cmake_minimum_required(VERSION 3.5)`
2. `src/EdgeDrawer.h` top → `#define _USE_MATH_DEFINES` (fixes missing `M_PI` on MSVC)
3. `pybind11` submodule → checkout `v2.13.6` (fixes Python 3.12 C-API incompatibility)
