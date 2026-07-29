from utils.env_setup import setup_env
setup_env()

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from libs.inference.road_segmentation import load_pidnet, predict_road, apply_road_mask
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_segmentation import split_left_right_lines
from libs.inference.lane_fitting import inner_chain_points, refine_inner_points, lane_curve
from libs.inference.pitch_estimation import estimate_pitch_from_curves
from libs.visualization.pitch_visualization import gt_pitch_profile
from libs.visualization.profile_mae_visualization import plot_profile_mae
import traceback

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device                    = config["model"]["device"]
model_name                = config["model"]["model_name"]
weight_path               = config["model"]["weight_path"]
image_batch_path          = config["input"]["image_batch_path"]
resize_size               = tuple(config["input"]["resize_size"])
min_segment_length_near   = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far    = config["line_segmentation"]["min_segment_length_far"]
min_slope                 = config["lane_segmentation"]["min_slope"]
lane_band_tolerance       = config["lane_segmentation"]["lane_band_tolerance"]
track_bands               = config["lane_segmentation"].get("track_bands", 16)
num_samples               = config["lane_fitting"]["num_samples"]
samples_per_meter         = config["lane_fitting"].get("samples_per_meter")
f_x                       = config["pitch_estimation"]["f_x"]
f_y                       = config["pitch_estimation"]["f_y"]
w_real                    = config["pitch_estimation"]["w_real"]
camera_height             = config["pitch_estimation"].get("camera_height")
camera_offset_m           = config["pitch_estimation"].get("camera_forward_offset", 0.0)
pitch_method              = config["pitch_estimation"].get("method", "windowed")
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]
measurements_csv          = config["csv_io"]["measurements_csv"]
problem_csv               = config["csv_io"]["problem_csv"]
problem_mae_threshold     = config["csv_io"]["problem_mae_threshold"]

IMG_FMT = "{:06d}.png"


def _compute_mae(pitch_curve, frame_id, measurements):
    """Return profile MAE (float) or None if degenerate or no valid GT samples."""
    if pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0:
        return None
    zs = pitch_curve["z_samples"]
    ps = pitch_curve["pitch_samples"]
    gt = gt_pitch_profile(measurements, frame_id, zs, camera_offset_m)
    valid = ~np.isnan(gt)
    if not valid.any():
        return None
    return round(float(np.abs(ps[valid] - gt[valid]).mean()), 4)


def main():
    df = pd.read_csv(input_csv)
    measurements = pd.read_csv(measurements_csv)
    model = load_pidnet(model_name, weight_path, device)

    n_total         = len(df)
    z_vis_min_col   = [pd.NA] * n_total
    z_vis_max_col   = [pd.NA] * n_total
    profile_mae_col = [pd.NA] * n_total
    n_skip_pipeline = 0
    skip_frame_ids  = []

    image_batch_dir = Path(image_batch_path)
    n_missing_image = 0

    for i, row in enumerate(df.itertuples(index=False)):
        frame_id   = int(row.frame_id)
        image_path = image_batch_dir / IMG_FMT.format(frame_id)

        if not image_path.exists():
            n_missing_image += 1
            continue

        try:
            # 1. Road segmentation
            resized_image, pred_mask = predict_road(model, str(image_path), device, resize_size)
            masked_road = apply_road_mask(resized_image, pred_mask)

            # 2. Line segmentation
            segments = detect_lines_with_elsed(
                masked_road, min_segment_length_near, min_segment_length_far
            )

            # 3. Lane segmentation
            inner_left, inner_right = split_left_right_lines(
                segments, resized_image.width, min_slope,
                resized_image.height, lane_band_tolerance, track_bands=track_bands,
                f_x=f_x, f_y=f_y, camera_height=camera_height, w_real=w_real,
            )

            # 4. Lane fitting — inner-chain points → continuous lane curves
            left_curve  = lane_curve(refine_inner_points(
                resized_image, inner_chain_points(inner_left,  True),  True))
            right_curve = lane_curve(refine_inner_points(
                resized_image, inner_chain_points(inner_right, False), False))

            # 5. Pitch estimation — widths from curves → continuous pitch(z)
            pitch_curve = estimate_pitch_from_curves(
                left_curve, right_curve, f_x, f_y, resized_image.height, w_real,
                num_samples=num_samples, samples_per_meter=samples_per_meter,
                method=pitch_method)

            if pitch_curve["pitch_at"] is not None:
                z_vis_min_col[i] = round(float(pitch_curve["z_visible_min"]), 2)
                z_vis_max_col[i] = round(float(pitch_curve["z_visible_max"]), 2)
                mae = _compute_mae(pitch_curve, frame_id, measurements)
                if mae is not None:
                    profile_mae_col[i] = mae

        except Exception as e:
            skip_frame_ids.append(frame_id)
            n_skip_pipeline += 1
            print(f"  skip frame {frame_id}: {type(e).__name__}: {e}")

    df["z_visible_min"] = z_vis_min_col
    df["z_visible_max"] = z_vis_max_col
    df["profile_mae"]   = profile_mae_col
    df.to_csv(output_csv, index=False)

    print(f"pipeline_skip={n_skip_pipeline}, missing_image={n_missing_image}, total={n_total}")
    print(f"skipped frame ids: {skip_frame_ids}")

    mae = pd.to_numeric(df["profile_mae"], errors="coerce").dropna()
    if len(mae):
        print(f"profile MAE over {len(mae)} frames: "
              f"mean={mae.mean():.4f}  median={mae.median():.4f}  "
              f"p90={mae.quantile(0.9):.4f}  max={mae.max():.4f}")

    problem_df = df.loc[mae[mae > problem_mae_threshold].index, ["frame_id", "profile_mae"]].copy()
    problem_df["image_path"] = problem_df["frame_id"].apply(
        lambda fid: str(image_batch_dir / IMG_FMT.format(int(fid)))
    )
    problem_df = problem_df.sort_values("profile_mae", ascending=False)
    Path(problem_csv).parent.mkdir(parents=True, exist_ok=True)
    problem_df.to_csv(problem_csv, index=False)
    print(f"problem frames (mae > {problem_mae_threshold}): {len(problem_df)} -> {problem_csv}")

    plot_path = plot_profile_mae(df, save_path="outputs/profile_mae.png")
    print(f"MAE plot: {plot_path}")


if __name__ == "__main__":
    main()
