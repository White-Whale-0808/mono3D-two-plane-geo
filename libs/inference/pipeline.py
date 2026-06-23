"""
Inference pipeline — unified lane segmentation with perspective-adaptive ROI.
Stages
------
1. Road segmentation   (PIDNet)
2. ELSED line detection
3. Lane segmentation   split_left_right_lines (ROI-based, innermost)
4. Piecewise linear fit
5. Pitch estimation
"""

import cv2
import numpy as np
import pyelsed

from libs.inference.road_segmentation import predict_road, apply_road_mask
from libs.inference.lane_segmentation import split_left_right_lines
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_fitting      import (
    collect_points_from_segments, piecewise_linear_fit, compute_lane_widths,
)
from libs.inference.pitch_estimation  import estimate_pitch_from_widths

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def infer_one(
    model, image_path, device, resize_size,
    min_slope, min_segment_length_near, min_segment_length_far, lane_band_tolerance,
    extra_points_per_segment, num_bands, num_samples,
    f_x, f_y, w_real,
    *,
    camera_height: float = None,
    return_debug: bool = False,
):
    """Run the full pipeline on a single image.

    Parameters
    ----------
    return_debug
        If True, also return a dict with intermediate values (n_segments).
    """
    # 1. road segmentation
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)

    # 2. ELSED line detection
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)

    # 3. lane segmentation (per-band innermost selection)
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, min_slope,
        resized_image.height, lane_band_tolerance, num_bands=num_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height, w_real=w_real,
    )

    # 4. lane fitting
    left_points  = collect_points_from_segments(inner_left,  extra_points_per_segment)
    right_points = collect_points_from_segments(inner_right, extra_points_per_segment)
    left_fits    = piecewise_linear_fit(left_points,  num_bands)
    right_fits   = piecewise_linear_fit(right_points, num_bands)

    # 5. pitch estimation
    #    pitch_per_depth : [(z_center, pitch_deg), ...] per-depth-band curve
    #    two_plane       : continuous hinge model — pitch_near_deg /
    #                      pitch_far_deg / z_knee / z_visible_min/max / pitch_at(z)
    widths          = compute_lane_widths(left_fits, right_fits, num_samples)
    pitch_per_depth = estimate_pitch_from_widths(widths, f_x, f_y, resized_image.height, w_real)

    result = {
        "pitch_per_depth": pitch_per_depth,   # per-depth-band Theil-Sen
    }
    if return_debug:
        result["debug"] = {
            "n_segments": int(len(segments)),
            "n_left": len(inner_left),
            "n_right": len(inner_right),
            "n_width_samples": int(len(widths)),
        }
    return result
