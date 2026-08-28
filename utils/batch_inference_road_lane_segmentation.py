from utils.env_setup import setup_env
setup_env()

import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from libs.inference.road_segmentation import load_pidnet
from libs.inference.pipeline import infer_one
from libs.inference.pitch_estimation import NearfieldWidthCalibrator
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
nearfield_w_real          = config["pitch_estimation"].get("nearfield_w_real", False)
gt_height_source          = config.get("ground_truth", {}).get("height_source", "auto")
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]
measurements_csv          = config["csv_io"]["measurements_csv"]
problem_csv               = config["csv_io"]["problem_csv"]
problem_mae_threshold     = config["csv_io"]["problem_mae_threshold"]

IMG_FMT = "{:06d}.png"


def _tagged(path, tag):
    """foo.csv + tag 'x' -> foo_x.csv（tag 為空則原樣返回）"""
    if not tag:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{tag}{p.suffix}"))


def _parse_args(argv=None):
    """CLI overrides for the config defaults.

    Everything defaults to the config, so a bare run is unchanged. The
    overrides exist so one acceptance sweep can cover several routes and
    both sides of a toggle without editing the config between runs (and
    without a second copy of the pipeline — see infer_one's comment).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=None,
                    help="資料集目錄（含 images/ 與 measurements.csv）；"
                         "省略則用 config 的路徑")
    ap.add_argument("--nearfield", dest="nearfield", action="store_true",
                    default=None, help="強制開啟近場 w_real 自標定")
    ap.add_argument("--no-nearfield", dest="nearfield", action="store_false",
                    help="強制關閉近場 w_real 自標定")
    ap.add_argument("--tag", default="",
                    help="輸出檔名後綴，連跑多組才不會互相覆蓋")
    return ap.parse_args(argv)


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


def main(argv=None):
    args = _parse_args(argv)
    # CLI 覆寫（沒給就是 config 的值）
    if args.dataset:
        in_csv   = str(args.dataset / "measurements.csv")
        meas_csv = in_csv
        image_dir_str = str(args.dataset / "images")
    else:
        in_csv, meas_csv = input_csv, measurements_csv
        image_dir_str = image_batch_path
    use_nearfield = (nearfield_w_real if args.nearfield is None
                     else args.nearfield)
    out_csv  = _tagged(output_csv, args.tag)
    prob_csv = _tagged(problem_csv, args.tag)

    df = pd.read_csv(in_csv)
    gt = load_profile_gt(meas_csv, camera_offset_m=camera_offset_m,
                         camera_height=camera_height,
                         height_source=gt_height_source)
    print(f"GT source: {gt.describe()}")
    model = load_pidnet(model_name, weight_path, device)

    n_total         = len(df)
    z_vis_min_col   = [pd.NA] * n_total
    z_vis_max_col   = [pd.NA] * n_total
    profile_mae_col = [pd.NA] * n_total
    w_real_used_col = [pd.NA] * n_total
    n_skip_pipeline = 0
    skip_frame_ids  = []

    # 近場自標定：一條路線一個 calibrator（θ0 閘門 + 沿用上一有效值），
    # config 的 w_real 降級為 fallback；關閉時行為與舊版完全相同
    calibrator = None
    if use_nearfield:
        calibrator = NearfieldWidthCalibrator(
            f_x, f_y, resize_size[0], camera_height, w_real)
        print("nearfield w_real self-calibration: ON")

    image_batch_dir = Path(image_dir_str)
    n_missing_image = 0

    for i, row in enumerate(df.itertuples(index=False)):
        frame_id   = int(row.frame_id)
        image_path = image_batch_dir / IMG_FMT.format(frame_id)

        if not image_path.exists():
            n_missing_image += 1
            continue

        if calibrator is not None and hasattr(row, "collect_dist_m"):
            calibrator.advance_to(float(row.collect_dist_m))

        try:
            # Full pipeline (road seg → ELSED → paint gate → tracking →
            # fitting + evidence/depth truncation → pitch); the batch runner
            # only needs the pitch curve, so it goes through infer_one and
            # never re-implements the stages (that duplication silently ran
            # the pre-WWH-15 pipeline here once already).
            result = infer_one(
                model, str(image_path), device, resize_size,
                min_segment_length_near, min_segment_length_far,
                num_samples, f_x, f_y, w_real, camera_height,
                samples_per_meter=samples_per_meter,
                track_bands=track_bands, method=pitch_method,
                w_real_calibrator=calibrator)
            pitch_curve = result["pitch_curve"]
            w_real_used_col[i] = round(float(result["w_real_used"]), 4)

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
    if use_nearfield:
        df["w_real_used"] = w_real_used_col
    df.to_csv(out_csv, index=False)

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
    Path(prob_csv).parent.mkdir(parents=True, exist_ok=True)
    problem_df.to_csv(prob_csv, index=False)
    print(f"problem frames (mae > {problem_mae_threshold}): {len(problem_df)} -> {prob_csv}")

    # 圖檔帶資料集名，連續跑多份資料集才不會互相覆蓋（CSV 仍照 config 走）
    dataset_dir = Path(meas_csv).parent
    plot_path = plot_profile_mae(
        df, save_path=_tagged(f"outputs/profile_mae_{dataset_dir.name}.png",
                              args.tag),
        title=f"Per-frame profile MAE — {dataset_dir.name}")
    print(f"MAE plot: {plot_path}")

    if (dataset_dir / "road_profile.csv").exists():
        route_png = _tagged(f"outputs/route_profile_{dataset_dir.name}.png",
                            args.tag)
        print("route profile plot: "
              f"{plot_route_profile(dataset_dir, df, save_path=route_png)}")


if __name__ == "__main__":
    main()
