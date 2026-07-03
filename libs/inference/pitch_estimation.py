import numpy as np
from scipy.stats import theilslopes
from scipy.interpolate import UnivariateSpline


def estimate_pitch_from_widths(widths, f_x, f_y, image_height, w_real,
                               min_profile_range_m: float = 3.0,
                               min_valid_range_m: float = 0.5,
                               z_cap_m: float = 45.0,
                               resid_mad_k: float = 5.0,
                               s: float = None,
                               k: int = 3,
                               n_pitch_samples: int = 200):
    """Estimate a continuous pitch(z) curve from per-band lane widths.

    Preprocessing (IQR width filter → depth/Y_3d → depth cap → sort →
    global Theil-Sen MAD residual filter) is unchanged from the banded version.
    A weighted UnivariateSpline on Y(z) replaces the per-band Theil-Sen loop;
    analytical differentiation gives a smooth, continuous pitch(z).

    Physical weights: w_i = 1/z_i^2 because depth uncertainty scales as z^2
    (z = f·W/width, so dz ∝ z²·dwidth). Far points are naturally down-weighted,
    making the spline smooth at range without hard truncation.

    Parameters
    ----------
    widths : np.ndarray, shape (N, 2)
        (y_pixel, pixel_width) pairs from compute_lane_widths*.
    s : float, optional
        Spline smoothing factor (UnivariateSpline `s`). Defaults to len(depths)
        after preprocessing. Increase to suppress far-range noise further.
    k : int
        Spline degree (default 3 = cubic). Auto-reduced when too few points.
    n_pitch_samples : int
        Number of uniformly spaced z samples in the returned arrays.

    Returns
    -------
    dict with keys:
        pitch_at      : callable z -> pitch_deg, clamped to [z_visible_min, z_visible_max]
        z_samples     : 1-D ndarray of n_pitch_samples depth values
        pitch_samples : corresponding pitch angles (deg), continuous
        z_visible_min : float
        z_visible_max : float
    On degenerate/short input pitch_at is None and the sample arrays are empty.
    """
    _empty = {
        "pitch_at": None,
        "z_samples": np.array([]),
        "pitch_samples": np.array([]),
        "z_visible_min": np.nan,
        "z_visible_max": np.nan,
    }

    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return _empty

    # IQR outlier filter on pixel width
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    valid = (w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)
    widths = widths[valid]
    if len(widths) == 0:
        return _empty

    depths = f_x * w_real / widths[:, 1]
    center_y = image_height / 2
    Y_3d = -depths * (widths[:, 0] - center_y) / f_y

    # Depth cap: z ~ 1/pixel_width, so spurious few-pixel widths near the
    # vanishing point explode to unphysical depths.
    in_range = depths <= z_cap_m
    if in_range.sum() >= 2:
        depths, Y_3d = depths[in_range], Y_3d[in_range]

    # Sort by depth
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    Y_3d   = Y_3d[sort_idx]

    # Robust residual filter: global Theil-Sen fit, drop points > resid_mad_k
    # robust sigmas. Catches off-plane junk (kerbs, crosswalks) that the
    # width-IQR filter cannot see.
    if len(depths) >= 4:
        fit = theilslopes(Y_3d, depths)
        resid = Y_3d - (fit.intercept + fit.slope * depths)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-9:
            keep = np.abs(resid - med) <= resid_mad_k * 1.4826 * mad
            if keep.sum() >= 2:
                depths, Y_3d = depths[keep], Y_3d[keep]

    z_range = float(depths[-1] - depths[0]) if len(depths) >= 2 else 0.0
    if z_range < min_valid_range_m or len(depths) < 2:
        return _empty

    z_vis_min = float(depths[0])
    z_vis_max = float(depths[-1])
    z_samps = np.linspace(z_vis_min, z_vis_max, n_pitch_samples)

    # Short range: too little depth coverage for a multi-knot spline —
    # fall back to a single global Theil-Sen slope (constant pitch).
    if z_range < min_profile_range_m or len(depths) < k + 1:
        res = theilslopes(Y_3d, depths)
        pitch_const = float(np.degrees(np.arctan(res.slope)))
        return {
            "pitch_at": lambda z, p=pitch_const: p,
            "z_samples": z_samps,
            "pitch_samples": np.full(n_pitch_samples, pitch_const),
            "z_visible_min": z_vis_min,
            "z_visible_max": z_vis_max,
        }

    # Aggregate duplicate depths so UnivariateSpline gets a strictly
    # monotone x array (multiple y_pixels can map to the same z).
    unique_depths, inv = np.unique(depths, return_inverse=True)
    unique_Y = np.array([Y_3d[inv == i].mean() for i in range(len(unique_depths))])
    depths, Y_3d = unique_depths, unique_Y

    # Physical weights: far-depth uncertainty ∝ z², so weight by 1/z²
    # → far points get lower influence → spline is naturally smoother at range.
    weights = 1.0 / np.clip(depths, 1e-6, None) ** 2
    s_eff   = float(len(depths)) if s is None else float(s)
    k_eff   = min(k, len(depths) - 1)

    spl  = UnivariateSpline(depths, Y_3d, w=weights, k=k_eff, s=s_eff)
    dspl = spl.derivative()

    def pitch_at(z):
        z_c = float(np.clip(z, z_vis_min, z_vis_max))
        return float(np.degrees(np.arctan(float(dspl(z_c)))))

    pitch_samps = np.degrees(np.arctan(dspl(z_samps)))

    return {
        "pitch_at": pitch_at,
        "z_samples": z_samps,
        "pitch_samples": pitch_samps,
        "z_visible_min": z_vis_min,
        "z_visible_max": z_vis_max,
    }
