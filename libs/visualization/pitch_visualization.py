"""Pitch profile visualization: predicted continuous pitch(z) vs distance-aligned GT.

x-axis: distance ahead of the vehicle (m)
y-axis: road pitch (deg), RELATIVE to the plane the vehicle currently sits on

GT alignment
------------
measurements.csv gives, per frame, the slope of the plane the vehicle is ON
(``gt_pitch_deg``) and the cumulative travelled distance (``collect_dist_m``).
The road surface d meters ahead of frame i is the plane the vehicle itself
will stand on once its cumulative distance reaches collect_dist_m[i] + d, so

    gt_profile_i(d) = gt_pitch_deg[j(d)] - gt_pitch_deg[i]
    j(d) = argmin_j | collect_dist_m[j] - (collect_dist_m[i] + d) |

Distances beyond the recorded drive yield NaN.
"""
import numpy as np
import pandas as pd

from libs.inference.pitch_estimation import back_project_widths


def gt_pitch_profile(measurements: pd.DataFrame, frame_id: int, distances):
    """Vehicle-relative GT pitch at each distance ahead (NaN if out of range)."""
    df = measurements.sort_values("collect_dist_m").reset_index(drop=True)
    row = df[df["frame_id"] == frame_id]
    if row.empty:
        raise ValueError(f"frame_id {frame_id} not in measurements")
    d0 = float(row["collect_dist_m"].iloc[0])
    p0 = float(row["gt_pitch_deg"].iloc[0])

    dist_arr  = df["collect_dist_m"].to_numpy()
    pitch_arr = df["gt_pitch_deg"].to_numpy()

    targets = d0 + np.atleast_1d(np.asarray(distances, dtype=float))
    # 最近鄰查找：dist_arr 已排序，searchsorted 找插入點後比較左右鄰
    idx   = np.clip(np.searchsorted(dist_arr, targets), 1, len(dist_arr) - 1)
    left  = dist_arr[idx - 1]
    right = dist_arr[idx]
    j = np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx)

    out = pitch_arr[j] - p0
    out[targets > dist_arr[-1] + 0.5] = np.nan
    return out


def gt_height_profile(measurements, frame_id, distances, camera_height,
                      step: float = 0.5):
    """GT road height Y(d) in the ego/camera frame at each distance ahead.

    Integrates tan(vehicle-relative GT pitch) over distance (trapezoid on a
    `step`-metre grid), anchored at Y(0) = -camera_height — the road surface
    directly below the camera. Distances beyond the recorded drive yield NaN.
    """
    d_req = np.atleast_1d(np.asarray(distances, dtype=float))
    d_max = float(np.max(d_req))
    grid = np.arange(0.0, d_max + step, step)
    slope = np.tan(np.radians(gt_pitch_profile(measurements, frame_id, grid)))
    seg = 0.5 * (slope[1:] + slope[:-1]) * np.diff(grid)
    y = -camera_height + np.concatenate(([0.0], np.cumsum(seg)))

    valid = ~np.isnan(y)
    if not valid.any():
        return np.full_like(d_req, np.nan)
    last = np.where(valid)[0][-1]
    return np.interp(d_req, grid[:last + 1], y[:last + 1], right=np.nan)


def plot_y3d_profile(frame_id, widths, pitch_curve, measurements,
                     f_x, f_y, image_height, w_real, camera_height,
                     max_dist=None, save_path=None, debug=False):
    """Plot predicted (z, Y_3d) height points vs the GT-derived height profile.

    Shows, in the ego/camera frame (road below camera at -camera_height):
      - black • : predict height — points that survived the estimator's
                  filters (subsampled to ``max_points`` for display)
      - orange — : fitted height curve Y(z) (spline / Theil-Sen line)
      - green — : GT height profile integrated from gt_pitch_deg
      - grey -- : -camera_height reference (road directly below the camera)
    With ``debug=True`` one extra layer appears:
      - grey ×  : raw back-projected width samples (pre-filter)
    A vertical offset between prediction and GT indicates a calibration /
    width-scale error (invisible in the pitch plot); shape mismatch indicates
    data or smoothing problems.

    Parameters
    ----------
    frame_id : int
    widths : np.ndarray, shape (N, 2) or None
        Raw (y_pixel, pixel_width) pairs from compute_lane_widths; only used
        when ``debug=True``.
    pitch_curve : dict
        Output of estimate_pitch_from_widths — uses ``z_points``, ``y_points``,
        ``z_samples``, ``y_samples``.
    measurements : pd.DataFrame or None
        Loaded measurements.csv; pass None to skip the GT curve.
    camera_height : float or None
        GT anchor height; the GT curve is skipped when None.
    max_dist : float, optional
        x-axis limit. Defaults to 1.5× the visible range.
    save_path : str, optional
        Output PNG path. Defaults to outputs/y3d_profile_{frame_id:06d}.png.
    debug : bool
        Also draw raw pre-filter samples.

    Returns
    -------
    str — path of the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_pts = pitch_curve.get("z_points", np.array([]))
    y_pts = pitch_curve.get("y_points", np.array([]))

    max_points = 80
    if len(z_pts) > max_points:
        idx = np.linspace(0, len(z_pts) - 1, max_points).round().astype(int)
        z_pts, y_pts = z_pts[idx], y_pts[idx]

    if max_dist is None:
        z_hi = pitch_curve.get("z_visible_max", np.nan)
        if np.isnan(z_hi):
            z_hi = float(z_pts.max()) if len(z_pts) else 30.0
        max_dist = max(z_hi * 1.5, 15)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if debug:
        z_raw, y_raw = back_project_widths(widths, f_x, f_y, image_height, w_real)
        if len(z_raw):
            ax.scatter(z_raw, y_raw, marker="x", s=18, color="0.65",
                       label=f"raw samples ({len(z_raw)})", zorder=2)
    z_samps = pitch_curve.get("z_samples", np.array([]))
    y_samps = pitch_curve.get("y_samples", np.array([]))
    if len(z_samps):
        ax.plot(z_samps, y_samps, color="tab:orange", lw=2,
                label="fitted height (spline)", zorder=4)
    if len(z_pts):
        ax.scatter(z_pts, y_pts, s=10, color="black", alpha=0.45,
                   label="predict height", zorder=5)

    if measurements is not None and camera_height is not None:
        d_grid = np.linspace(0, max_dist, 400)
        gt_y = gt_height_profile(measurements, frame_id, d_grid, camera_height)
        ax.plot(d_grid, gt_y, color="tab:green", lw=2,
                label="GT height (integrated)", zorder=4)
        ax.axhline(-camera_height, color="0.8", lw=1, ls="--",
                   label=f"-camera_height ({-camera_height:.2f} m)", zorder=1)

    ax.set_xlim(0, max_dist)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("Y_3d (m)")
    ax.set_title(f"frame {frame_id}: road height profile")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is None:
        save_path = f"outputs/y3d_profile_{frame_id:06d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_pitch_profile(frame_id, pitch_curve, measurements,
                       max_dist=None, save_path=None):
    """Plot continuous predicted pitch(z) vs GT.

    Parameters
    ----------
    frame_id : int
    pitch_curve : dict
        Output of estimate_pitch_from_widths — must contain keys
        ``z_samples``, ``pitch_samples``, ``z_visible_min``, ``z_visible_max``.
    measurements : pd.DataFrame
        Loaded from measurements.csv.
    max_dist : float, optional
        x-axis limit. Defaults to 1.5× the visible range.
    save_path : str, optional
        Output PNG path. Defaults to outputs/pitch_profile_{frame_id:06d}.png.

    Returns
    -------
    str — path of the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_samps     = pitch_curve.get("z_samples",     np.array([]))
    pitch_samps = pitch_curve.get("pitch_samples", np.array([]))
    has_pred    = len(z_samps) > 0

    if has_pred:
        z_lo = pitch_curve["z_visible_min"]
        z_hi = pitch_curve["z_visible_max"]
    else:
        z_lo, z_hi = 0, 30

    if max_dist is None:
        max_dist = max(z_hi * 1.5, 15)

    d_grid = np.linspace(0, max_dist, 200)
    gt     = gt_pitch_profile(measurements, frame_id, d_grid)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d_grid, gt, color="tab:blue", lw=2, label="GT (distance-aligned)")
    if has_pred:
        ax.plot(z_samps, pitch_samps, color="tab:orange", lw=2,
                label="Predicted (continuous spline)")

    ax.set_xlabel("distance ahead (m)")
    ax.set_ylabel("road pitch relative to ego plane (deg)")
    ax.set_title(f"Frame {frame_id:06d}: predicted pitch profile vs GT")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path is None:
        save_path = f"outputs/pitch_profile_{frame_id:06d}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
