"""Per-frame profile_mae visualization for batch inference output.

Typical usage (from batch_inference_road_lane_segmentation.py):

    from libs.visualization.profile_mae_visualization import plot_profile_mae
    plot_profile_mae(df, save_path="outputs/profile_mae.png")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_profile_mae(
    df: pd.DataFrame,
    save_path: str = "outputs/profile_mae.png",
    title: str = "Per-frame profile MAE",
) -> str:
    """Plot profile_mae per frame and save to PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``frame_id`` and ``profile_mae`` columns
        (as produced by batch_inference_road_lane_segmentation).
        Rows where profile_mae is NaN (pipeline skipped / no GT) are
        drawn as vertical grey ticks at the top to make gaps visible.
    save_path : str
        Output PNG path.
    title : str
        Figure title.

    Returns
    -------
    str — path of the saved figure.
    """
    mae = pd.to_numeric(df["profile_mae"], errors="coerce")
    frame_ids = df["frame_id"].to_numpy()

    valid_mask = ~mae.isna()
    x_valid = frame_ids[valid_mask]
    y_valid = mae[valid_mask].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.scatter(x_valid, y_valid, color="tab:blue", s=18, label="profile MAE (deg)")

    if len(y_valid):
        mean_mae   = y_valid.mean()
        median_mae = np.median(y_valid)
        p90_mae    = np.percentile(y_valid, 90)
        ax.axhline(mean_mae,   color="tab:orange", lw=1.2, ls="--", label=f"mean {mean_mae:.4f}°")
        ax.axhline(median_mae, color="tab:green",  lw=1.2, ls=":",  label=f"median {median_mae:.4f}°")
        ax.axhline(p90_mae,    color="tab:red",    lw=1.0, ls="-.", label=f"p90 {p90_mae:.4f}°")

    ax.set_xlabel("frame id")
    ax.set_ylabel("profile MAE (deg)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
