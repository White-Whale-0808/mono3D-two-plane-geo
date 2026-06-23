import time
from pathlib import Path
import pandas as pd
import yaml
from libs.inference.road_segmentation import load_pidnet, predict_road, apply_road_mask
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_segmentation import split_left_right_lines
from libs.visualization.lane_visualization import draw_lane_lines, create_overlay, draw_line_segments
from libs.inference.lane_fitting import collect_points_from_segments, piecewise_linear_fit, compute_lane_widths
from libs.visualization.lane_visualization import draw_piecewise_fits
from libs.inference.pitch_estimation import estimate_pitch_from_widths
from libs.visualization.pitch_visualization import plot_pitch_profile, gt_pitch_profile

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
alpha = config["visualization"]["alpha"]
save_path = config["visualization"]["save_path"]
num_bands = config["lane_fitting"]["num_bands"]
num_samples = config["lane_fitting"]["num_samples"]
extra_points_per_segment = config["lane_fitting"]["extra_points_per_segment"]
f_x = config["pitch_estimation"]["f_x"]
f_y = config["pitch_estimation"]["f_y"]
w_real = config["pitch_estimation"]["w_real"]
camera_height    = config["pitch_estimation"].get("camera_height")
measurements_csv = config["csv_io"]["measurements_csv"]


def compute_pitch_mae(pitch_per_depth, frame_id, measurements):
    """Compute MAE between predicted per-depth pitch and distance-aligned GT.

    For each predicted (z, pitch) band, looks up the GT pitch at that distance
    ahead using the cumulative-distance alignment in measurements.csv, then
    returns the mean absolute error across all valid bands.

    Returns None if no valid GT samples exist (e.g. frame near end of dataset).
    """
    import numpy as np
    zs = [z for z, _ in pitch_per_depth]
    ps = np.array([p for _, p in pitch_per_depth])
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
        lane_band_tolerance, num_bands=num_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height, w_real=w_real,
    )
    t3 = time.perf_counter()

    """
    Lane fitting
    """
    left_points = collect_points_from_segments(inner_left, extra_points_per_segment)
    right_points = collect_points_from_segments(inner_right, extra_points_per_segment)
    left_fits = piecewise_linear_fit(left_points, num_bands)
    right_fits = piecewise_linear_fit(right_points, num_bands)
    widths = compute_lane_widths(left_fits, right_fits, num_samples)
    t4 = time.perf_counter()
    
    """
    Pitch estimation
    """
    pitch_per_depth = estimate_pitch_from_widths(widths, f_x, f_y, resized_image.height, w_real)
    for z, theta in pitch_per_depth:
        print(f"  z={z:.1f}m  pitch={theta:.2f}°")
    t5 = time.perf_counter()

    print(f"road segmentation:   {(t1-t0)*1000:.1f} ms")
    print(f"line segmentation:      {(t2-t1)*1000:.1f} ms")
    print(f"lane segmentation:   {(t3-t2)*1000:.1f} ms")
    print(f"lane fitting:        {(t4-t3)*1000:.1f} ms")
    print(f"pitch estimation:    {(t5-t4)*1000:.1f} ms")

    if Path(measurements_csv).exists():
        frame_id = int(Path(image_path).stem)
        measurements = pd.read_csv(measurements_csv)
        mae = compute_pitch_mae(pitch_per_depth, frame_id, measurements)
        if mae is not None:
            print(f"pitch MAE:           {mae:.4f}°")
        pitch_plot_path = save_path.replace(".png", "_pitch_profile.png")
        plot_pitch_profile(frame_id, pitch_per_depth, measurements,
                           save_path=pitch_plot_path)
        print(f"pitch profile:       {pitch_plot_path}")

    overlay_save_path = save_path.replace(".png", "_overlay.png")
    create_overlay(resized_image, pred_mask, alpha, overlay_save_path)
    draw_line_save_path = save_path.replace(".png", "_line_segments.png")
    draw_line_segments(resized_image, segments, draw_line_save_path)
    draw_lane_save_path = save_path.replace(".png", "_lanes.png")
    draw_lane_lines(resized_image, inner_left, inner_right, draw_lane_save_path)
    draw_piecewise_fits_save_path = save_path.replace(".png", "_lane_fits.png")
    draw_piecewise_fits(resized_image, left_fits, right_fits, widths, draw_piecewise_fits_save_path)
if __name__ == "__main__":
    main()
