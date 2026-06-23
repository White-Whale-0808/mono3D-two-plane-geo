from utils.env_setup import setup_env
setup_env()

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from libs.inference.road_segmentation import load_pidnet
from libs.inference.pipeline import infer_one
from libs.visualization.pitch_visualization import gt_pitch_profile
from libs.visualization.profile_mae_visualization import plot_profile_mae
import traceback

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device                    = config["model"]["device"]
model_name                = config["model"]["model_name"]
weight_path               = config["model"]["weight_path"]
image_batch_path          = config["input"]["image_batch_path"]
image_batch_size          = config["input"]["image_batch_size"]
resize_size               = tuple(config["input"]["resize_size"])
min_segment_length_near   = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far    = config["line_segmentation"]["min_segment_length_far"]
min_slope                 = config["lane_segmentation"]["min_slope"]
lane_band_tolerance       = config["lane_segmentation"]["lane_band_tolerance"]
num_bands                 = config["lane_fitting"]["num_bands"]
num_samples               = config["lane_fitting"]["num_samples"]
extra_points_per_segment  = config["lane_fitting"]["extra_points_per_segment"]
f_x                       = config["pitch_estimation"]["f_x"]
f_y                       = config["pitch_estimation"]["f_y"]
w_real                    = config["pitch_estimation"]["w_real"]
camera_height             = config["pitch_estimation"].get("camera_height")
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]
measurements_csv          = config["csv_io"]["measurements_csv"]
profile_samples           = config["csv_io"]["profile_samples"]

IMG_FMT = "{:06d}.png"


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
        frame_id = int(row.frame_id)
        image_path = image_batch_dir / IMG_FMT.format(frame_id)

        if not image_path.exists():
            n_missing_image += 1
            continue

        try:
            result = infer_one(
                model, str(image_path),
                device=device, resize_size=resize_size,
                min_slope=min_slope,
                min_segment_length_near=min_segment_length_near,
                min_segment_length_far=min_segment_length_far,
                lane_band_tolerance=lane_band_tolerance,
                extra_points_per_segment=extra_points_per_segment,
                num_bands=num_bands, num_samples=num_samples,
                f_x=f_x, f_y=f_y, w_real=w_real,
                camera_height=camera_height,
            )

            pitch_per_depth = result["pitch_per_depth"]
            if pitch_per_depth:
                zs = [z for z, _ in pitch_per_depth]
                ps = np.array([p for _, p in pitch_per_depth])
                z_vis_min_col[i] = round(min(zs), 2)
                z_vis_max_col[i] = round(max(zs), 2)

                # Profile MAE: |per-band predicted pitch - distance-aligned GT|
                gt = gt_pitch_profile(measurements, frame_id, zs)
                valid = ~np.isnan(gt)
                if valid.any():
                    profile_mae_col[i] = round(
                        float(np.abs(ps[valid] - gt[valid]).mean()), 4
                    )

        except Exception:
            skip_frame_ids.append(frame_id)
            n_skip_pipeline += 1
            print("-" * 40)
            print(f"bug: inference pipeline skipped frame id {frame_id} due to error")
            traceback.print_exc()

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

    plot_path = plot_profile_mae(df, save_path="outputs/profile_mae.png")
    print(f"profile MAE plot saved to {plot_path}")


if __name__ == "__main__":
    main()
