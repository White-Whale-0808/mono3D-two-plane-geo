"""
Inference pipeline — unified lane segmentation with perspective-adaptive ROI.
Stages
------
1. Road segmentation   (PIDNet)
2. ELSED line detection
3. Lane segmentation   split_left_right_lines (ROI-based, innermost)
4. Lane fitting        inner-chain points → continuous lane curves
5. Pitch estimation    widths from curves → continuous spline pitch(z)
"""

import cv2
import numpy as np
import pyelsed

from libs.inference.road_segmentation import predict_road, apply_road_mask
from libs.inference.lane_segmentation import split_left_right_lines
from libs.inference.line_segmentation import detect_lines_with_elsed
from libs.inference.lane_fitting      import inner_chain_points, refine_inner_points, lane_curve
from libs.inference.pitch_estimation  import estimate_pitch_from_curves

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def infer_one(
    model, image_path, device, resize_size,
    min_slope, min_segment_length_near, min_segment_length_far, lane_band_tolerance,
    num_samples,
    f_x, f_y, w_real,
    *,
    camera_height: float = None,
    samples_per_meter: float = None,
    track_bands: int = 16,
    return_debug: bool = False,
):
    """Run the full pipeline on a single image.

    Parameters
    ----------
    return_debug
        If True, also return a dict with intermediate values.
    """
    # 1. road segmentation
    resized_image, pred_mask = predict_road(model, image_path, device, resize_size)
    masked_road = apply_road_mask(resized_image, pred_mask)

    # 2. ELSED line detection
    segments = detect_lines_with_elsed(masked_road, min_segment_length_near, min_segment_length_far)

    # 3. lane segmentation (per-band innermost selection)
    inner_left, inner_right = split_left_right_lines(
        segments, resized_image.width, min_slope,
        resized_image.height, lane_band_tolerance, track_bands=track_bands,
        f_x=f_x, f_y=f_y, camera_height=camera_height, w_real=w_real,
    )

    # 4. lane fitting — per-row inner-envelope chain (w_real is inner-edge
    # to inner-edge; the tracker keeps whole marking groups for evidence),
    # then a continuous gap-bridged curve per side
    left_curve  = lane_curve(refine_inner_points(
        resized_image, inner_chain_points(inner_left,  True),  True))
    right_curve = lane_curve(refine_inner_points(
        resized_image, inner_chain_points(inner_right, False), False))

    # 5. pitch estimation — widths from the curves → continuous spline pitch(z)
    pitch_curve = estimate_pitch_from_curves(
        left_curve, right_curve, f_x, f_y, resized_image.height, w_real,
        num_samples=num_samples, samples_per_meter=samples_per_meter)

    result = {"pitch_curve": pitch_curve}
    if return_debug:
        degenerate = pitch_curve["pitch_at"] is None or len(pitch_curve["z_samples"]) == 0
        result["debug"] = {
            "n_segments": int(len(segments)),
            "n_left": len(inner_left),
            "n_right": len(inner_right),
            "n_width_samples": int(len(pitch_curve["widths"])),
            "pitch_degenerate": degenerate,
        }
    return result
