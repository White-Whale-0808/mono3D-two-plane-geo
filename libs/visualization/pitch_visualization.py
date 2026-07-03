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
    max_d     = dist_arr[-1]

    out = []
    for d in np.atleast_1d(distances):
        target = d0 + float(d)
        if target > max_d + 0.5:
            out.append(np.nan)
            continue
        j = int(np.argmin(np.abs(dist_arr - target)))
        out.append(pitch_arr[j] - p0)
    return np.asarray(out)


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
