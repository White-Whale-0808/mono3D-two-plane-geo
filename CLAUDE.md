# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monocular road pitch angle estimation using two-plane geometry. The pipeline processes single camera images through 5 sequential stages to estimate the road's pitch angle in degrees.

## Setup

```bash
# Requires Python 3.12, uses uv as package manager
cp .env.example .env        # Machine-specific paths (CARLA wheel, OpenCV dirs)
uv sync                     # Install dependencies
python scripts/setup_elsed.py  # Build local patched pyelsed (ELSED C++ extension)
```

PIDNet weights must be placed in `pidnet_pretrained_model/PIDNet_L_Cityscapes_test.pt`.

## Common Commands

```bash
# Single image inference with visualization
python -m utils.inference_road_lane_segmentation

# Batch inference (CSV in/out)
python -m utils.batch_inference_road_lane_segmentation

# CARLA real-time test (requires running CARLA server)
python carla_module/realtime_test.py [--host HOST] [--port PORT] [--map MAP]

# Plot GT vs predicted pitch
python main.py
```

All inference config lives in `config/inference_road_lane_segmentation.yaml`.

## Pipeline Architecture

Entry point: `libs/inference/pipeline.py::infer_one()`

```
road_segmentation → line_segmentation → lane_segmentation → lane_fitting → pitch_estimation
```

| Stage | File | What it does |
|---|---|---|
| road_segmentation | `libs/inference/road_segmentation.py` | PIDNet-L semantic segmentation → binary road mask |
| line_segmentation | `libs/inference/line_segmentation.py` | ELSED line detection with adaptive length threshold (perspective-aware) |
| lane_segmentation | `libs/inference/lane_segmentation_up_hile.py` or `_down_hile.py` | Split lines into left/right lanes, select innermost |
| lane_fitting | `libs/inference/lane_fitting.py` | Piecewise linear fit in horizontal bands, compute lane widths |
| pitch_estimation | `libs/inference/pitch_estimation.py` | Inverse perspective → depth, Theil-Sen regression → pitch angle |

Two lane segmentation variants exist for different road slopes:
- **up_hile**: Selects innermost lanes, simple perspective handling
- **down_hile**: Adds adaptive ROI filtering for downhill perspective distortion

## Critical Conventions

- **`setup_env()` must be called before any C extension imports** (cv2, pyelsed, carla). See `utils/env_setup.py`.
- **Image format**: Pipeline expects RGB throughout. Use `cv2.cvtColor` to BGR only for `cv2.imwrite()`.
- **Coordinate system**: OpenCV convention (origin top-left, y-down). Left lane has negative slope, right lane positive.
- **`resize_size` in config is `[height, width]`**, but PIL expects `(width, height)` — the swap is handled in `predict_road()`.
- **CARLA overrides `f_y = f_x`** (square pixels) in `carla_module/realtime_test.py::load_config()`.

## Config Structure (YAML)

Config sections map 1:1 to pipeline stages:
- `road_segmentation` — (currently empty, PIDNet uses argmax not threshold)
- `line_segmentation` — `min_segment_length_near`, `min_segment_length_far`
- `lane_segmentation` — `min_slope`, `lane_band_tolerance`
- `lane_fitting` — `extra_points_per_segment`, `num_bands`, `num_samples`
- `pitch_estimation` — `f_x`, `f_y`, `w_real`
