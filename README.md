# mono3D-two-plane-geo

Monocular road pitch angle estimation using two-plane geometry. The pipeline processes single-camera images through five sequential stages to estimate the road's pitch angle in degrees, using only a pretrained semantic segmentation model and classical geometry — no depth sensor required.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [1. Single-Image Inference (with visualization)](#1-single-image-inference-with-visualization)
  - [2. Batch Inference on a Dataset](#2-batch-inference-on-a-dataset)
  - [3. Evaluate: GT vs Predicted Pitch](#3-evaluate-gt-vs-predicted-pitch)
  - [4. Error Analysis](#4-error-analysis)
  - [5. CARLA Real-Time Test](#5-carla-real-time-test)
- [Full Workflow (End-to-End Execution Order)](#full-workflow-end-to-end-execution-order)
- [Configuration Reference](#configuration-reference)
- [Lane Segmentation Variants](#lane-segmentation-variants)
- [Critical Conventions](#critical-conventions)

---

## How It Works

The core entry point is `libs/inference/pipeline.py::infer_one()`, which chains five stages:

```
road_segmentation → line_segmentation → lane_segmentation → lane_fitting → pitch_estimation
```

| Stage | File | What it does |
|---|---|---|
| road_segmentation | `libs/inference/road_segmentation.py` | PIDNet-L semantic segmentation → binary road mask (Cityscapes class 0) |
| line_segmentation | `libs/inference/line_segmentation.py` | ELSED line detection with adaptive length threshold (perspective-aware) |
| lane_segmentation | `libs/inference/lane_segmentation_up_hile.py` or `_down_hile.py` | Split lines into left/right, select innermost (or outermost for downhill) |
| lane_fitting | `libs/inference/lane_fitting.py` | Piecewise linear fit per horizontal band, compute lane widths |
| pitch_estimation | `libs/inference/pitch_estimation.py` | Inverse perspective → depth → 3D Y coords → Theil-Sen regression → pitch angle |

**Pitch formula:**

```
depth    = f_x × w_real / width_pixel
Y_3d     = -depth × (y_pixel - image_height/2) / f_y
pitch    = arctan( TheilSen_slope(Y_3d ~ depth) )
```

---

## Project Structure

```
mono3D-two-plane-geo/
├── config/
│   ├── inference_road_lane_segmentation.yaml   # Main config for all inference
│   └── convert_metadata_to_gt.yaml             # Config for GT conversion
├── libs/
│   ├── inference/
│   │   ├── pipeline.py                         # Core pipeline: infer_one()
│   │   ├── road_segmentation.py                # PIDNet loader + masking
│   │   ├── line_segmentation.py                # ELSED line detection (perspective-aware)
│   │   ├── lane_segmentation_up_hile.py        # Lane split for uphill roads
│   │   ├── lane_segmentation_down_hile.py      # Lane split for downhill roads
│   │   ├── lane_fitting.py                     # Piecewise linear fit + width sampling
│   │   └── pitch_estimation.py                 # Geometric pitch estimation
│   └── visualization/
│       └── lane_visualization.py               # Draw masks, lanes, fits
├── carla_module/
│   ├── realtime_test.py                        # CARLA real-time inference loop
│   ├── carla_road_segmentation.py              # PIDNet adapter for PIL input
│   └── carla_visualization.py                  # CARLA display rendering
├── utils/
│   ├── env_setup.py                            # Must be called before C extension imports
│   ├── inference_road_lane_segmentation.py     # Single-image inference + save visualizations
│   ├── batch_inference_road_lane_segmentation.py  # Batch inference → CSV output
│   ├── convert_metadata_to_gt.py               # Convert raw CARLA metadata to GT labels
│   ├── plot_frameId_and_pitch.py               # Plot GT vs predicted pitch
│   └── analyze_error/
│       ├── find_top_error.py                   # Export top-K highest error frames
│       └── analyze_interval_error.py           # Compute MAE per frame interval
├── scripts/
│   └── setup_elsed.py                          # Clone + patch ELSED C++ extension
├── pidnet_models/                              # PIDNet model definitions
├── pidnet_pretrained_model/                    # Pretrained weights (not tracked in git)
│   └── PIDNet_L_Cityscapes_test.pt
├── outputs/                                    # Inference results, CSVs, plots
├── .env.example                                # Template for machine-specific paths
└── pyproject.toml                              # Dependencies (managed by uv)
```

---

## Prerequisites

- Python **3.12** (exact version required)
- [`uv`](https://github.com/astral-sh/uv) as the package manager
- A C++ compiler (for building the ELSED extension):
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
  - **Windows**: Visual Studio Build Tools with MSVC
  - **Linux**: `gcc` / `g++`
- **Windows only**: A system-installed OpenCV SDK (not the PyPI wheel) — path set via `.env`

For CARLA real-time testing, additionally:
- A running CARLA simulator server (tested with 0.9.16)
- The matching CARLA Python `.whl` installed manually

---

## Installation

**Step 1 — Copy and fill in the environment file**

```bash
cp .env.example .env
```

Edit `.env` with your machine-specific paths. On macOS/Linux, you can leave the Windows-only fields empty:

```dotenv
# Windows only — paths to OpenCV DLLs and CMake config for building pyelsed
OPENCV_BIN_PATH=
OPENCV_DIR=

# Path to CARLA .whl (required only for CARLA real-time test)
CARLA_WHL_PATH=
```

**Step 2 — Patch and prepare the ELSED C++ source**

This must be done before `uv sync`, because `pyproject.toml` references the local `elsed_src/` directory as the source for `pyelsed`.

```bash
python scripts/setup_elsed.py
```

This script clones the ELSED repo into `elsed_src/`, pins pybind11 to v2.13.6 (required for Python 3.12), and applies Windows/MSVC compatibility patches.

**Step 3 — Install all dependencies (this also compiles pyelsed)**

```bash
uv sync
```

**Step 4 — Place pretrained PIDNet weights**

Download `PIDNet_L_Cityscapes_test.pt` and place it at:

```
pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt
```

**Step 5 — (Optional) Install CARLA Python package**

Only needed for real-time CARLA testing. Install the `.whl` that matches your CARLA server version:

```bash
uv pip install $CARLA_WHL_PATH
```

---

## Running the Project

> **Important:** All commands must be run from the **project root directory**. The config file and relative paths in the code depend on this.

### 1. Single-Image Inference (with visualization)

Runs the full 5-stage pipeline on one image and saves intermediate visualizations to `outputs/`.

```bash
python -m utils.inference_road_lane_segmentation
```

Set the target image in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_path: "inference_datasets/solid_line/up_hile/images/000201.png"
```

Output files saved to `outputs/`:

| File | Content |
|---|---|
| `result_overlay.png` | Road mask overlaid on image |
| `result_line_segments.png` | All detected ELSED line segments |
| `result_lanes.png` | Inner left/right lane lines |
| `result_lane_fits.png` | Piecewise fits + sampled width points |

Inference timing per stage is printed to stdout.

---

### 2. Batch Inference on a Dataset

Runs inference on every frame listed in a CSV and writes `pred_deg` back to the same CSV.

**Before running**, you need a GT CSV with at least a `frame_id` column. See [Full Workflow](#full-workflow-end-to-end-execution-order) below for how to generate it.

Set paths in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_batch_path: "inference_datasets/solid_line/up_hile/images"
csv_io:
  input_dir: "outputs/metadata_gt.csv"
  output_dir: "outputs/metadata_gt.csv"   # overwrites in-place
```

```bash
python -m utils.batch_inference_road_lane_segmentation
```

Frames that fail (e.g., no lane detected) are skipped and logged. A summary of skip count is printed at the end.

---

### 3. Evaluate: GT vs Predicted Pitch

Plots `gt_pitch_deg` and `pred_deg` against `frame_id` from the CSV produced in step 2.

```bash
python main.py
```

Output: `outputs/pitch_plot.png`

---

### 4. Error Analysis

These utilities read from the output CSV of batch inference.

**Find the top-10 highest-error frames:**

```bash
python -m utils.analyze_error.find_top_error
```

Output: `outputs/top_errors.csv`

**Compute MAE by frame interval:**

```bash
python -m utils.analyze_error.analyze_interval_error
```

Prints MAE for the hard-coded frame intervals (196–410 and 410–500). Edit the script to change intervals.

---

### 5. CARLA Real-Time Test

Requires a running CARLA server. Spawns a vehicle, attaches an RGB camera, and runs the full inference pipeline on each frame in real time, overlaying estimated and ground-truth pitch.

```bash
python carla_module/realtime_test.py [--host HOST] [--port PORT] [--map MAP] [--timeout SEC]
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | CARLA server address |
| `--port` | `2000` | CARLA server port |
| `--map` | `Town03` | CARLA map name |
| `--timeout` | `20.0` | Connection timeout (seconds) |

Press **`q`** to quit. On exit, the script destroys all spawned CARLA actors automatically.

> **Note:** The CARLA camera uses square pixels (FOV=90°, 1024×512), so `f_y` is overridden to equal `f_x` inside `load_config()` regardless of what the YAML says.

---

## Full Workflow (End-to-End Execution Order)

This is the complete pipeline for collecting data from CARLA and evaluating pitch estimation accuracy.

```
Step 0: Setup (one-time)
─────────────────────────────────────────────────────
cp .env.example .env                              # fill in machine-specific paths
python scripts/setup_elsed.py                     # patch ELSED C++ source
uv sync                                           # install deps + compile pyelsed
uv pip install $CARLA_WHL_PATH                    # (if using CARLA)

Step 1: Collect raw metadata from CARLA
─────────────────────────────────────────────────────
# Run your CARLA data collection script to produce:
#   inference_datasets/solid_line/up_hile/images/   ← frame images (000001.png, ...)
#   inference_datasets/solid_line/up_hile/metadata_filtered.csv
#                                                  ← columns: frame_id, pitch (vehicle transform pitch)

Step 2: Convert raw metadata to ground truth
─────────────────────────────────────────────────────
# Edit config/convert_metadata_to_gt.yaml:
#   slope_deg: 12.25          ← true road slope in degrees
#   input_dir: "inference_datasets/solid_line/up_hile/metadata_filtered.csv"
#   output_dir: "outputs/metadata_gt.csv"
python -m utils.convert_metadata_to_gt
# Produces outputs/metadata_gt.csv with column gt_pitch_deg = slope_deg - pitch

Step 3: Run batch inference
─────────────────────────────────────────────────────
# Edit config/inference_road_lane_segmentation.yaml:
#   image_batch_path: "inference_datasets/solid_line/up_hile/images"
#   csv_io.input_dir / output_dir: "outputs/metadata_gt.csv"
python -m utils.batch_inference_road_lane_segmentation
# Adds pred_deg column to outputs/metadata_gt.csv

Step 4: Plot results
─────────────────────────────────────────────────────
python main.py
# Saves outputs/pitch_plot.png

Step 5: Analyze errors (optional)
─────────────────────────────────────────────────────
python -m utils.analyze_error.find_top_error
python -m utils.analyze_error.analyze_interval_error
```

---

## Configuration Reference

All inference parameters live in `config/inference_road_lane_segmentation.yaml`.

```yaml
model:
  device: "cpu"                 # "cpu" or "cuda"
  model_name: "pidnet_l"        # PIDNet variant
  weight_path: "pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt"

input:
  image_path: "..."             # single-image inference target
  image_batch_path: "..."       # directory for batch inference
  image_batch_size: 305
  resize_size: [512, 1024]      # [height, width] — PIL will swap internally

line_segmentation:
  min_segment_length_near: 65   # min segment length (px) at bottom of image
  min_segment_length_far: 0     # min segment length (px) at top of image
                                # threshold is linearly interpolated by mid-y

lane_segmentation:
  min_slope: 0.3                # discard segments with |slope| < this
  lane_band_tolerance: 10       # max x_at_bottom deviation (px) to merge into same lane

lane_fitting:
  extra_points_per_segment: 10  # interpolated points added per segment before fitting
  num_bands: 3                  # number of horizontal bands for piecewise fitting
  num_samples: 20               # y-samples for lane width computation

pitch_estimation:
  f_x: 512                      # horizontal focal length (px, after resize)
  f_y: 455                      # vertical focal length (px, after resize)
  w_real: 3.5                   # real-world lane width (meters)

visualization:
  alpha: 0.4
  save_path: "outputs/result.png"

csv_io:
  input_dir: "outputs/metadata_gt.csv"
  output_dir: "outputs/metadata_gt.csv"
```

GT conversion config (`config/convert_metadata_to_gt.yaml`):

```yaml
slope_deg: 12.25                # true road slope used to compute gt_pitch_deg
io:
  input_dir: "inference_datasets/solid_line/up_hile/metadata_filtered.csv"
  output_dir: "outputs/metadata_gt.csv"
```

---

## Lane Segmentation Variants

Two implementations exist for different road conditions. Switch by changing the import in `libs/inference/pipeline.py`:

| Variant | File | Use case | Selection strategy |
|---|---|---|---|
| `up_hile` | `lane_segmentation_up_hile.py` | Uphill / flat roads | Picks the **innermost** lane lines; uses `x_at_bottom` to rank |
| `down_hile` | `lane_segmentation_down_hile.py` | Downhill roads | Picks the **outermost** lane lines; adds adaptive ROI filtering based on `mid_y` to handle perspective distortion |

`up_hile` is the default used in `pipeline.py` and `utils/inference_road_lane_segmentation.py`.

---

## Critical Conventions

- **`setup_env()` must be the very first call** in any entry-point script, before any `import cv2`, `import pyelsed`, or `import carla`. It loads `.env` and registers OpenCV DLL paths on Windows. See `utils/env_setup.py`.
- **Image format**: The pipeline expects **RGB** throughout. Only convert to BGR with `cv2.cvtColor` immediately before `cv2.imwrite()`.
- **Coordinate system**: OpenCV convention — origin at top-left, y increases downward. Left lane lines have **negative** slope, right lane lines have **positive** slope.
- **`resize_size` is `[height, width]`** in the YAML, but PIL's `image.resize()` expects `(width, height)`. The swap is handled inside `predict_road()` — do not swap it again.
- **CARLA overrides `f_y = f_x`** (square pixels) inside `carla_module/realtime_test.py::load_config()`. The YAML value for `f_y` is ignored in CARLA mode.
