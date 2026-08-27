# mono3D-two-plane-geo

Monocular road pitch estimation from a single camera. The pipeline runs five sequential stages over one image and returns a **continuous pitch(z) curve** — the road's pitch angle in degrees as a function of depth ahead of the camera — using only a pretrained semantic segmentation model and classical projective geometry. No depth sensor required.

> **On the name:** the project started with an explicit near/far two-plane model. That model was replaced in WWH-9 by a continuous pitch(z) curve; there is no longer a knee or a pair of planes. The repository name is historical.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Calibration and the Evidence Guards](#calibration-and-the-evidence-guards)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
  - [1. Single-Image Inference (with visualization)](#1-single-image-inference-with-visualization)
  - [2. Batch Inference on a Dataset](#2-batch-inference-on-a-dataset)
  - [3. Unit Tests](#3-unit-tests)
  - [4. CARLA Real-Time Test (currently broken)](#4-carla-real-time-test-currently-broken)
- [Ground Truth](#ground-truth)
- [Configuration Reference](#configuration-reference)
- [Critical Conventions](#critical-conventions)

---

## How It Works

The pipeline lives in `libs/inference/pipeline.py::infer_one()`, which chains the five stages:

```
road_segmentation → line_segmentation → lane_segmentation → lane_fitting → pitch_estimation
```

| Stage | File | What it does |
|---|---|---|
| road_segmentation | `libs/inference/road_segmentation.py` | PIDNet-L semantic segmentation → binary road mask (Cityscapes class 0) |
| line_segmentation | `libs/inference/line_segmentation.py` | ELSED line detection with a perspective-aware length threshold (interpolated by segment mid-y) |
| lane_segmentation | `libs/inference/lane_segmentation.py` | Near-to-far continuity tracking: seed the innermost lines at the bottom, track upward band by band with a local line model, re-seed past junctions |
| lane_fitting | `libs/inference/lane_fitting.py` | Inner-chain extraction → sub-pixel edge refinement on the **unmasked** image → one continuous gap-bridged lane curve per side |
| pitch_estimation | `libs/inference/pitch_estimation.py` | Sample lane widths from the two curves, inverse-perspective them to depth, then a continuous pitch(z) |

**Geometry:**

```
depth  z    = f_x × w_real / w_px          # w_px = inner-edge-to-inner-edge pixel width
Y_3d        = -z × (y_pixel - cy) / f_y    # cy = image_height / 2
pitch(z)    = arctan( slope of Y_3d over a local z-window )
```

The default estimator is `windowed`: for each sampled depth it takes a Theil-Sen slope over a `±max(1 m, 0.15·z)` window, which makes the spatial resolution of the output explicit. `method: spline` switches to a global weighted `UnivariateSpline` with a Theil-Sen MAD prefilter.

**Callers.** `utils/batch_inference_road_lane_segmentation.py` calls `infer_one()` directly. `utils/inference_road_lane_segmentation.py` deliberately inlines the same stages so it can draw each intermediate product — if you change `pipeline.py`, that second copy needs the same change.

---

## Calibration and the Evidence Guards

`f_x`, `f_y`, `w_real` and `camera_height` are **required**. Under the flat-ground pinhole model every lateral threshold has an exact pixel form at row `y`, so association tolerance, seed window, slope gates and model memory are all *derived* rather than hand-tuned. Three evidence guards ride on the same projection:

| Guard | Where | What it rejects |
|---|---|---|
| Paint segment gate | `paint_evidence.filter_paint_segments`, before tracking | ELSED segments that are not a bright ridge of bounded width — shadow boundaries, kerbs. The tracker then re-seeds on the true painted line |
| Evidence-break truncation | `paint_evidence.truncate_at_evidence_break`, after refinement | The tail of a chain once the photometric "is this paint?" test fails for a sustained run — this is what crest occlusion looks like |
| Depth-continuity truncation | `lane_fitting.truncate_at_depth_jump` | A jump in paired-row depth that exceeds both a continuity gate and 1.5× the local-plane extrapolation. Real paint beyond a crest is a *disconnected* section and must never be joined to the near chain |

Their thresholds are module constants with the derivations in the docstrings, not config values.

> A hand-tuned fallback for un-calibrated cameras (`min_slope` / `lane_band_tolerance` / `roi_near`) used to sit alongside this and was removed on 2026-08-27: every caller supplied the full calibration, so it was a second implementation that nothing ran and no test covered. Its thresholds were fitted to one dataset anyway, so a genuinely different camera would need them re-derived rather than reused — which is what the geometry path does on its own.

---

## Project Structure

```
mono3D-two-plane-geo/
├── config/
│   ├── inference_road_lane_segmentation.yaml   # Main config for all inference
│   └── train_road_segmentation.yaml            # Config for the (legacy) Resnet101 training path
├── libs/
│   ├── inference/
│   │   ├── pipeline.py                         # infer_one(): the five stages
│   │   ├── geometry.py                         # CameraGeometry: the pinhole model shared by the stages
│   │   ├── road_segmentation.py                # PIDNet loader + masking
│   │   ├── line_segmentation.py                # ELSED line detection (perspective-aware)
│   │   ├── lane_segmentation.py                # Near-to-far continuity tracking
│   │   ├── paint_evidence.py                   # Photometric "is this actually paint?" guards
│   │   ├── lane_fitting.py                     # Inner chain → sub-pixel refine → lane curves
│   │   └── pitch_estimation.py                 # Width sampling → depth → continuous pitch(z)
│   ├── road_profile_gt.py                      # Measured road-profile ground truth
│   └── visualization/
│       ├── lane_visualization.py               # Masks, segments, lanes, curves, per-step dumps
│       ├── pitch_visualization.py              # Per-frame pitch / Y_3d profile plots
│       ├── profile_mae_visualization.py        # Per-frame MAE plot (batch)
│       └── route_profile_visualization.py      # Whole-route terrain profile plot (batch)
├── carla_module/
│   ├── get_carlaDataset.py                     # Dataset collector (images + measurements + road profile)
│   ├── pick_route.py                           # Top-down route picking tool
│   ├── project_lane_gt.py                      # Project lane GT into the image
│   ├── verify_carla_geometry.py                # Calibration checks against the simulator
│   ├── realtime_test.py                        # Real-time inference loop — CURRENTLY BROKEN, see below
│   ├── carla_road_segmentation.py              # PIDNet adapter for PIL input
│   └── carla_visualization.py                  # CARLA display rendering
├── utils/
│   ├── env_setup.py                            # Must be called before any C extension import
│   ├── inference_road_lane_segmentation.py     # Single-image inference + all visualizations
│   └── batch_inference_road_lane_segmentation.py  # Batch inference → CSV + plots
├── tests/
│   ├── synthetic.py                            # Synthetic pinhole scenes with analytically-known answers
│   ├── test_geometry.py                        # The projection model
│   ├── test_pitch_estimation.py                # Metric stage against known grades
│   ├── test_paint_evidence.py                  # Paint guards' edge cases
│   └── test_depth_jump.py                      # Depth-continuity guard's edge cases
├── debug/                                      # Diagnostic and prototype scripts (gitignored, not part of the pipeline)
├── scripts/
│   └── setup_elsed.py                          # Clone + patch the ELSED C++ extension
├── docs/papers/                                # Reference papers and design drawios
├── pidnet_models/                              # PIDNet model definitions
├── pidnet_pretrained_model/                    # Pretrained weights (not tracked in git)
│   └── PIDNet_L_Cityscapes_test.pt
├── outputs/                                    # Inference results, CSVs, plots
├── main.py                                     # Thin wrapper around single-image inference
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

For CARLA data collection, additionally:
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

# Path to CARLA .whl (required only for CARLA work)
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

**Step 5 — (Optional) Install the CARLA Python package**

Only needed for data collection or real-time testing. Install the `.whl` that matches your CARLA server version:

```bash
uv pip install $CARLA_WHL_PATH
```

---

## Running the Project

> **Important:** All commands must be run from the **project root directory**. The config file and the relative paths in the code depend on this.

### 1. Single-Image Inference (with visualization)

Runs the five stages on one image and saves every intermediate product to `outputs/`.

```bash
python -m utils.inference_road_lane_segmentation
# python main.py is a thin wrapper around exactly this
```

Set the target image in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_path: "inference_datasets/<dataset>/images/000500.png"
```

Output files (base name comes from `visualization.save_path`):

| File | Content |
|---|---|
| `result_overlay.png` | Road mask overlaid on the image |
| `result_line_segments.png` | All detected ELSED line segments |
| `result_lanes_seg.png` | The tracked inner left/right lane segments |
| `result_lane_fits.png` | The two continuous lane curves |
| `result_pitch_profile.png` | Predicted pitch(z) against GT |
| `result_y3d_profile.png` | Predicted road height Y_3d(z) against GT |
| `outputs/lane_fit_step/` | Per-step dump of the lane-fitting stage |
| `outputs/pitch_est_step/` | Per-step dump of the pitch-estimation stage |

Per-stage timing, the GT source, and the frame's pitch MAE are printed to stdout. GT is read from the image's own dataset, so no separate GT file has to be prepared.

---

### 2. Batch Inference on a Dataset

Runs inference on every frame in a dataset's `measurements.csv` and writes the results back out.

```bash
python -m utils.batch_inference_road_lane_segmentation
```

Set the paths in `config/inference_road_lane_segmentation.yaml`:

```yaml
input:
  image_batch_path: "inference_datasets/<dataset>/images"
csv_io:
  measurements_csv: "inference_datasets/<dataset>/measurements.csv"
  output_dir: "outputs/measurements.csv"
  problem_csv: "outputs/problem.csv"
  problem_mae_threshold: 2.0
```

Produces:

| Output | Content |
|---|---|
| `csv_io.output_dir` | The input CSV plus `z_visible_min`, `z_visible_max`, `profile_mae` per frame |
| `outputs/problem.csv` | Frames with `profile_mae` above `problem_mae_threshold`, worst first |
| `outputs/profile_mae_<dataset>.png` | Per-frame MAE over the route |
| `outputs/route_profile_<dataset>.png` | Whole-route terrain profile (analytic vs mesh height, their difference, per-frame MAE aligned to distance). Only when the dataset has a `road_profile.csv` |

The plot filenames carry the dataset name so that running several datasets back to back does not overwrite them; the CSV paths still follow the config.

Frames that produce no output are skipped and their ids printed. **A skipped frame is not necessarily a bug** — the method needs two inner lane edges, so intersections and unmarked crests legitimately give nothing, and the paint guards will abstain rather than measure a kerb.

---

### 3. Unit Tests

```bash
uv run --no-sync python -m pytest
```

> ⚠ Use `--no-sync`. A plain `uv run` (or `uv sync`, or `uv add`) re-resolves the environment and replaces the CUDA build of torch with the CPU wheel the lockfile pins. If that happens, restore it with:
> ```bash
> uv pip install "torch==2.10.0+cu126" "torchvision==0.25.0+cu126" --index-url https://download.pytorch.org/whl/cu126
> ```

The tests are pure geometry — no images, no weights, no GPU, under two seconds. They cover two things:

- **The projection model and the metric stage**, against synthetic pinhole scenes whose answer is known analytically (a road of constant grade must read back as that grade). `tests/synthetic.py` builds the scenes from the forward model the pipeline inverts.
- **The three evidence guards' edge cases**, each of which cost a full three-dataset sweep to find: the leading-drop exception, the run-length rule, the fixed bright-peak window, and the large-row-gap case that must *not* be read as a depth jump. The docstrings name the frames.

They do **not** measure accuracy. That is what the batch MAE sweep is for.

---

### 4. CARLA Real-Time Test (currently broken)

```bash
python carla_module/realtime_test.py [--host HOST] [--port PORT] [--map MAP] [--timeout SEC]
```

> ⚠ **This path does not currently run.** `realtime_test.py` and `carla_visualization.py` still import `collect_points_from_segments`, `piecewise_linear_fit`, `compute_lane_widths` and `fit_two_plane_model`, all of which were removed in WWH-7 / WWH-9. Reviving it means migrating to `lane_curve` / `sample_widths_from_curves` / `estimate_pitch_from_curves` and adding the three evidence guards. Two further notes for whoever does it: `realtime_test.py` mounts its camera at 2.4 m and overrides `camera_height`, and it overrides `f_y = f_x` because the CARLA camera has square pixels; and its `w_real` means inner-edge to inner-edge, same as everywhere else.

Data **collection** from CARLA (`carla_module/get_carlaDataset.py`, `pick_route.py`) is unaffected and works.

---

## Ground Truth

GT is not a pipeline stage — it is the reference the runners score against, and it lives in its own `ground_truth` config section.

A collected dataset carries two files: `measurements.csv` (per frame: the camera transform, vehicle distance, body pitch) and `road_profile.csv` (the road surface ahead, in world coordinates). `libs/road_profile_gt.py` projects the profile into the camera frame with `v = P_world − cam_world`, `z_gt = v·forward`, `h_gt = v·up` — no offset constant, no arc-length conversion, and it never goes through the vehicle's attitude.

`road_profile.csv` has two height columns and `ground_truth.height_source` picks between them:

| Value | Column | Meaning |
|---|---|---|
| `auto` (default) | `z_mesh` if present, else `z` | Falls back for pre-WWH-14 datasets |
| `analytic` | `z` | The OpenDRIVE analytic centreline: smooth, without the implemented surface detail |
| `mesh` | `z_mesh` | A downward ray-cast onto the road mesh at collection time |

`mesh` is the better reference: the collector's road surface deviates from the analytic centreline in a few localised sections and the camera demonstrably sees those deviations. Asking for `mesh` on a dataset that lacks the column raises rather than silently falling back. The runners print `GT source:` so you can see which one was used.

> ⚠ Do not re-litigate the analytic/mesh choice with **absolute-height** MAE — it is dominated by a per-frame constant offset. The evidence is written up in the `libs/road_profile_gt.py` module docstring and the config comments.

Datasets with no `road_profile.csv` fall back to a legacy GT that reconstructs the road ahead from the vehicle's own later body pitch.

---

## Configuration Reference

All inference parameters live in `config/inference_road_lane_segmentation.yaml`. The sections map 1:1 onto the pipeline stages, plus `ground_truth`, `visualization` and `csv_io`. The file itself carries long comments recording where each calibrated value came from; this is only a map.

```yaml
model:
  device: "cuda"                # "cpu" or "cuda"
  model_name: "pidnet_l"
  weight_path: "pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt"

input:
  image_path: "..."             # single-image inference target
  image_batch_path: "..."       # directory for batch inference
  resize_size: [512, 1024]      # [height, width] — PIL swaps internally

road_segmentation:              # no parameters: PIDNet-L uses argmax, and no
                                # morphological cleanup is applied to the mask

line_segmentation:
  min_segment_length_near: 65   # min segment length (px) at the bottom of the image
  min_segment_length_far: 0     # ... and at the top; interpolated linearly by mid-y

lane_segmentation:              # thresholds are derived from pitch_estimation's calibration
  track_bands: 16               # continuity-tracking band count (clamped to >= 16 internally)

lane_fitting:
  num_samples: 80               # width-sample fallback when samples_per_meter is unset
  samples_per_meter: 6          # geometry mode: z-uniform width-sample density, per metre of visible depth

pitch_estimation:
  f_x: 512                      # horizontal focal length (px, after resize)
  f_y: 455                      # vertical focal length (px, after resize)
  w_real: 3.25                  # inner-edge-to-inner-edge lane width (m) — see the warning below
  camera_height: 1.08           # camera mount height above the road (m); also enables geometry mode
  camera_forward_offset: 1.5    # camera mount offset ahead of the vehicle origin (m), for GT distance alignment
  method: windowed              # "windowed" (default) or "spline"

ground_truth:
  height_source: "auto"         # "auto" | "analytic" | "mesh"

visualization:
  alpha: 0.4
  save_path: "outputs/result.png"

csv_io:
  measurements_csv: "inference_datasets/<dataset>/measurements.csv"
  output_dir: "outputs/measurements.csv"
  problem_csv: "outputs/problem.csv"
  problem_mae_threshold: 2.0
```

> ⚠ **`w_real = 3.25` is specific to this road's marking layout** — double yellow on the left, single white on the right, on a 3.5 m lane. Because `w_real` is inner-edge to inner-edge, each side is inset from the boundary-centre width by a different amount (left 0.1875, right 0.0625, total 0.25). On the same 3.5 m lane: single+single → 3.375, double+single → **3.25**, double+double → 3.125. Re-derive it per road; do not carry 3.25 to another map or another lane. And determine it from height/geometry, never by minimising MAE — MAE trades the z scale against pitch error and its optimum is systematically pulled low.

---

## Critical Conventions

- **`setup_env()` must be the very first call** in any entry-point script, before any `import cv2`, `import pyelsed`, or `import carla`. It loads `.env` and registers OpenCV DLL paths on Windows. See `utils/env_setup.py`.
- **Image format**: the pipeline expects **RGB** throughout. Only convert to BGR with `cv2.cvtColor` immediately before `cv2.imwrite()`.
- **Coordinate system**: OpenCV convention — origin at top-left, y increases downward. Left lane lines have **negative** slope, right lane lines have **positive** slope.
- **`resize_size` is `[height, width]`** in the YAML, but PIL's `image.resize()` expects `(width, height)`. The swap is handled inside `predict_road()` — do not swap it again.
- **`w_real` is inner-edge to inner-edge**, not CARLA's `waypoint.lane_width` (which is boundary-centre to boundary-centre).
- **CARLA overrides `f_y = f_x`** (square pixels) and `camera_height = 2.4` inside `carla_module/realtime_test.py::load_config()`. The YAML values are ignored in that mode.
- **`pipeline.py` has a second copy**: `utils/inference_road_lane_segmentation.py` inlines the same stages to draw intermediates. Changes to the pipeline must be made in both.
