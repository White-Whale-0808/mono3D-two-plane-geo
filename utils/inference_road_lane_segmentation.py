from utils.env_setup import setup_env
setup_env()

import time
from pathlib import Path
import pandas as pd
import yaml
from libs.inference.road_segmentation import load_pidnet, predict_road, apply_road_mask
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_segmentation import split_left_right_lines
from libs.visualization.lane_visualization import draw_lane_lines, create_overlay, draw_line_segments, save_lane_fitting_steps
from libs.inference.lane_fitting import inner_chain_points, refine_inner_points, lane_curve
from libs.visualization.lane_visualization import draw_lane_curves
from libs.inference.pitch_estimation import estimate_pitch_from_curves
from libs.visualization.pitch_visualization import plot_pitch_profile, gt_pitch_profile, plot_y3d_profile, save_pitch_estimation_steps

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = config["model"]["device"]
model_name = config["model"]["model_name"]
weight_path = config["model"]["weight_path"]
image_path = config["input"]["image_path"]
resize_size = tuple(config["input"]["resize_size"])
min_segment_length_near = config["line_segmentation"]["min_segment_length_near"]
min_segment_length_far  = config["line_segmentation"]["min_segment_length_far"]
min_slope = config["lane_segmentation"]["min_slope"]
lane_band_tolerance = config["lane_segmentation"]["lane_band_tolerance"]
track_bands = config["lane_segmentation"].get("track_bands", 16)
alpha = config["visualization"]["alpha"]
save_path = config["visualization"]["save_path"]
num_samples = config["lane_fitting"]["num_samples"]
samples_per_meter = config["lane_fitting"].get("samples_per_meter")
f_x = config["pitch_estimation"]["f_x"]
f_y = config["pitch_estimation"]["f_y"]
w_real = config["pitch_estimation"]["w_real"]
camera_height    = config["pitch_estimation"].get("camera_height")
pitch_method     = config["pitch_estimation"].get("method", "windowed")
measurements_csv = config["csv_io"]["measurements_csv"]


def compute_pitch_mae(pitch_curve, frame_id, measurements):
    """Compute MAE between predicted continuous pitch(z) and distance-aligned GT."""
    import numpy as np
    if pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0:
        return None
    zs = pitch_curve["z_samples"]
    ps = pitch_curve["pitch_samples"]
    gt = gt_pitch_profile(measurements, frame_id, zs)
    valid = ~np.isnan(gt)
    if not valid.any():
        return None
    return float(np.abs(ps[valid] - gt[valid]).mean())


def main():
    """
    Road segementation
    """
    model = load_pidnet(model_name, weight_path, device)
    t0 = time.perf_counter()
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)
    t1 = time.perf_counter()

    """
    Line segementation
    """
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)
    t2 = time.perf_counter()

    """
    Lane segementation
    """
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, min_slope, resized_image.height,
        lane_band_tolerance, track_bands=track_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height, w_real=w_real,
    )
    t3 = time.perf_counter()

    """
    Lane fitting
    """
    left_points = refine_inner_points(
        resized_image, inner_chain_points(inner_left, True), True)
    right_points = refine_inner_points(
        resized_image, inner_chain_points(inner_right, False), False)
    left_curve = lane_curve(left_points)
    right_curve = lane_curve(right_points)
    t4 = time.perf_counter()

    """
    Pitch estimation
    """
    pitch_curve = estimate_pitch_from_curves(
        left_curve, right_curve, f_x, f_y, resized_image.height, w_real,
        num_samples=num_samples, samples_per_meter=samples_per_meter,
        method=pitch_method)
    widths = pitch_curve["widths"]
    zs, ps = pitch_curve["z_samples"], pitch_curve["pitch_samples"]
    if len(zs) == 0:
        print("pitch: (degenerate — no valid depth range)")
    else:
        step = max(1, len(zs) // 10)
        for z, theta in zip(zs[::step], ps[::step]):
            print(f"  z={z:.1f}m  pitch={theta:.2f}°")
    t5 = time.perf_counter()

    print(f"road segmentation:   {(t1-t0)*1000:.1f} ms")
    print(f"line segmentation:      {(t2-t1)*1000:.1f} ms")
    print(f"lane segmentation:   {(t3-t2)*1000:.1f} ms")
    print(f"lane fitting:        {(t4-t3)*1000:.1f} ms")
    print(f"pitch estimation:    {(t5-t4)*1000:.1f} ms")

    measurements, frame_id = None, None
    if Path(measurements_csv).exists():
        frame_id = int(Path(image_path).stem)
        measurements = pd.read_csv(measurements_csv)

        mae = compute_pitch_mae(pitch_curve, frame_id, measurements)
        if mae is not None:
            print(f"pitch MAE: {mae:.4f}°")

        pitch_plot_path = save_path.replace(".png", "_pitch_profile.png")
        plot_pitch_profile(frame_id, pitch_curve, measurements, save_path=pitch_plot_path)
        print(f"pitch profile: {pitch_plot_path}")

        y3d_plot_path = save_path.replace(".png", "_y3d_profile.png")
        plot_y3d_profile(frame_id, widths, pitch_curve, measurements,
                         f_x, f_y, resized_image.height, w_real, camera_height,
                         save_path=y3d_plot_path)
        print(f"y3d profile: {y3d_plot_path}")

    overlay_save_path = save_path.replace(".png", "_overlay.png")
    create_overlay(resized_image, pred_mask, alpha, overlay_save_path)
    draw_line_save_path = save_path.replace(".png", "_line_segments.png")
    draw_line_segments(resized_image, segments, draw_line_save_path)
    draw_lane_save_path = save_path.replace(".png", "_lanes_seg.png")
    draw_lane_lines(resized_image, inner_left, inner_right, draw_lane_save_path)
    draw_lane_curves_save_path = save_path.replace(".png", "_lane_fits.png")
    draw_lane_curves(resized_image, left_curve, right_curve, draw_lane_curves_save_path)

    steps_dir = save_lane_fitting_steps(resized_image, inner_left, inner_right,
                                        "outputs/lane_fit_step")
    print(f"lane fitting steps: {steps_dir}")
    steps_dir = save_pitch_estimation_steps(
        resized_image, left_curve, right_curve, pitch_curve,
        f_x, f_y, resized_image.height, w_real, "outputs/pitch_est_step",
        measurements=measurements, frame_id=frame_id)
    print(f"pitch estimation steps: {steps_dir}")

if __name__ == "__main__":
    main()
