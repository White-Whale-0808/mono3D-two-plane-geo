from utils.env_setup import setup_env
setup_env()

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from libs.inference.road_segmentation import load_pidnet
from libs.inference.pipeline import infer_one
from libs.road_profile_gt import load_profile_gt
from libs.visualization.profile_mae_visualization import plot_profile_mae
from libs.visualization.route_profile_visualization import plot_route_profile
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
track_bands               = config["lane_segmentation"].get("track_bands", 16)
num_samples               = config["lane_fitting"]["num_samples"]
samples_per_meter         = config["lane_fitting"].get("samples_per_meter")
f_x                       = config["pitch_estimation"]["f_x"]
f_y                       = config["pitch_estimation"]["f_y"]
w_real                    = config["pitch_estimation"]["w_real"]
camera_height             = config["pitch_estimation"].get("camera_height")
camera_offset_m           = config["pitch_estimation"].get("camera_forward_offset", 0.0)
pitch_method              = config["pitch_estimation"].get("method", "windowed")
gt_height_source          = config.get("ground_truth", {}).get("height_source", "auto")
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]
measurements_csv          = config["csv_io"]["measurements_csv"]
problem_csv               = config["csv_io"]["problem_csv"]
problem_mae_threshold     = config["csv_io"]["problem_mae_threshold"]

IMG_FMT = "{:06d}.png"


def _compute_mae(pitch_curve, frame_id, gt):
    """Return profile MAE (float) or None if degenerate or no valid GT samples."""
    if pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0:
        return None
    zs = pitch_curve["z_samples"]
    ps = pitch_curve["pitch_samples"]
    gt_deg = gt.pitch_at(frame_id, zs)
    valid = ~np.isnan(gt_deg)
    if not valid.any():
        return None
    return round(float(np.abs(ps[valid] - gt_deg[valid]).mean()), 4)


def main():
    df = pd.read_csv(input_csv)
    gt = load_profile_gt(measurements_csv, camera_offset_m=camera_offset_m,
                         camera_height=camera_height,
                         height_source=gt_height_source)
    print(f"GT source: {gt.describe()}")
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
            # Full pipeline (road seg → ELSED → paint gate → tracking →
            # fitting + evidence/depth truncation → pitch); the batch runner
            # only needs the pitch curve, so it goes through infer_one and
            # never re-implements the stages (that duplication silently ran
            # the pre-WWH-15 pipeline here once already).
            pitch_curve = infer_one(
                model, str(image_path), device, resize_size,
                min_segment_length_near, min_segment_length_far,
                num_samples, f_x, f_y, w_real, camera_height,
                samples_per_meter=samples_per_meter,
                track_bands=track_bands, method=pitch_method)["pitch_curve"]

            if pitch_curve["pitch_at"] is not None:
                z_vis_min_col[i] = round(float(pitch_curve["z_visible_min"]), 2)
                z_vis_max_col[i] = round(float(pitch_curve["z_visible_max"]), 2)
                mae = _compute_mae(pitch_curve, frame_id, gt)
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

    # 圖檔帶資料集名，連續跑多份資料集才不會互相覆蓋（CSV 仍照 config 走）
    dataset_dir = Path(measurements_csv).parent
    plot_path = plot_profile_mae(
        df, save_path=f"outputs/profile_mae_{dataset_dir.name}.png",
        title=f"Per-frame profile MAE — {dataset_dir.name}")
    print(f"MAE plot: {plot_path}")

    if (dataset_dir / "road_profile.csv").exists():
        print(f"route profile plot: {plot_route_profile(dataset_dir, df)}")


if __name__ == "__main__":
    main()
