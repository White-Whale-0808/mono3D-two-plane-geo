import numpy as np
from scipy.stats import theilslopes
from scipy.interpolate import UnivariateSpline

# Windowed-estimator defaults, exposed for the step visualization: the local
# Theil-Sen window is |z - zc| <= max(WINDOW_MIN_M, WINDOW_FRAC * zc).
WINDOW_FRAC = 0.15
WINDOW_MIN_M = 1.0


def back_project_widths(widths, f_x, f_y, image_height, w_real):
    """Back-project (y_pixel, pixel_width) pairs to camera-frame (z, Y_3d).

    Inverse perspective: z = f_x·w_real/width, Y_3d = -z·(y_pixel - cy)/f_y
    with cy = image_height/2. Non-positive widths are unphysical (lane fits
    crossing near the vanishing point) and are dropped; no other filtering.
    Canonical implementation shared by the estimator and the debug plots.

    Returns (z, Y_3d) 1-D arrays, empty on malformed/empty input.
    """
    widths = np.asarray(widths)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return np.array([]), np.array([])
    widths = widths[widths[:, 1] > 0]
    if len(widths) == 0:
        return np.array([]), np.array([])
    z = f_x * w_real / widths[:, 1]
    Y_3d = -z * (widths[:, 0] - image_height / 2) / f_y
    return z, Y_3d


def sample_widths_from_curves(left_curve, right_curve, num_samples, *,
                              f_x=None, w_real=None, samples_per_meter=None):
    """(y, width) samples over the y-overlap of two continuous lane curves.

    The curves come from lane_fitting.lane_curve: {"y": ascending, "x": ...},
    already gap-bridged. Width is a metric quantity (only meaningful through
    f_x / w_real), so sampling it belongs to this stage, not lane_fitting.

    Geometry mode (f_x and w_real given): dense candidate sweep → depth via
    z = f_x·w_real/width → resample uniformly in z (samples_per_meter, or
    num_samples when unset) so each meter of visible depth gets equal sample
    density. Legacy mode: y-uniform num_samples.
    """
    if left_curve is None or right_curve is None:
        return np.empty((0, 2))
    y_lo = max(left_curve["y"][0], right_curve["y"][0])
    y_hi = min(left_curve["y"][-1], right_curve["y"][-1])
    if not y_hi > y_lo:
        return np.empty((0, 2))

    def width_at(ys):
        return (np.interp(ys, right_curve["y"], right_curve["x"])
                - np.interp(ys, left_curve["y"], left_curve["x"]))

    if f_x is not None and w_real is not None:
        candidate_ys = np.linspace(y_lo, y_hi, 2000)
        ws = width_at(candidate_ys)
        valid = ws > 0
        if valid.sum() < 2:
            return np.empty((0, 2))
        ys_v, ws_v = candidate_ys[valid], ws[valid]
        zs = f_x * w_real / ws_v
        order = np.argsort(zs)
        zs_s, ys_s, ws_s = zs[order], ys_v[order], ws_v[order]
        if samples_per_meter is not None:
            # Scene-invariant density: n scales with the visible depth range,
            # capped at the candidate sweep resolution.
            z_range = zs_s[-1] - zs_s[0]
            n_samples = int(np.clip(np.ceil(z_range * samples_per_meter),
                                    2, len(candidate_ys)))
        else:
            n_samples = num_samples
        target_zs = np.linspace(zs_s[0], zs_s[-1], n_samples)
        target_ys = np.interp(target_zs, zs_s, ys_s)
        target_ws = np.interp(target_zs, zs_s, ws_s)
        return np.column_stack([target_ys, target_ws])

    sample_ys = np.linspace(y_lo, y_hi, num_samples)
    return np.column_stack([sample_ys, width_at(sample_ys)])


def estimate_pitch_from_curves(left_curve, right_curve, f_x, f_y, image_height,
                               w_real, *, num_samples, samples_per_meter=None,
                               method="windowed", **kwargs):
    """Metric stage entry point: two continuous lane curves in, pitch out.

    Samples lane widths from the curves, then runs the continuous pitch(z)
    estimator — `method="spline"` (weighted UnivariateSpline, global) or
    `method="windowed"` (local z-window Theil-Sen). Returns the estimator
    dict plus "widths" (the sampled (y, w) array, for visualization /
    Y_3d profiling).
    """
    widths = sample_widths_from_curves(
        left_curve, right_curve, num_samples,
        f_x=f_x, w_real=w_real, samples_per_meter=samples_per_meter)
    est = estimate_pitch_windowed if method == "windowed" else estimate_pitch_from_widths
    result = est(widths, f_x, f_y, image_height, w_real, **kwargs)
    result["widths"] = widths
    return result


def _empty_result():
    return {
        "pitch_at": None,
        "z_samples": np.array([]),
        "pitch_samples": np.array([]),
        "y_samples": np.array([]),
        "z_points": np.array([]),
        "y_points": np.array([]),
        "z_visible_min": np.nan,
        "z_visible_max": np.nan,
    }


def _preprocess_widths(widths, f_x, f_y, image_height, w_real, z_cap_m):
    """Shared width→(z, Y_3d) preprocessing: IQR width filter, back-projection,
    depth cap, sort by depth. Returns (depths, Y_3d), possibly empty."""
    widths = np.asarray(widths)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return np.array([]), np.array([])
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    widths = widths[(w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)]
    if len(widths) == 0:
        return np.array([]), np.array([])
    depths, Y_3d = back_project_widths(widths, f_x, f_y, image_height, w_real)
    if len(depths) == 0:
        return depths, Y_3d
    # Depth cap: z ~ 1/pixel_width, so spurious few-pixel widths near the
    # vanishing point explode to unphysical depths.
    in_range = depths <= z_cap_m
    if in_range.sum() >= 2:
        depths, Y_3d = depths[in_range], Y_3d[in_range]
    order = np.argsort(depths)
    return depths[order], Y_3d[order]


def estimate_pitch_windowed(widths, f_x, f_y, image_height, w_real,
                            min_valid_range_m: float = 0.5,
                            z_cap_m: float = 45.0,
                            window_frac: float = WINDOW_FRAC,
                            window_min_m: float = WINDOW_MIN_M,
                            min_window_points: int = 4,
                            n_pitch_samples: int = 200):
    """Continuous pitch(z) from local z-window Theil-Sen slopes.

    Explicitly-local alternative to the global spline: at each output depth
    z_c, pitch is the Theil-Sen slope of the (z, Y_3d) points within
    |z - z_c| <= max(window_min_m, window_frac·z_c). The window IS the
    spatial resolution of the profile — it grows with z for the same reason
    the spline down-weights far points (depth noise ∝ z²). There is NO
    global residual filter: the local median is robust by itself, and the
    global Theil-Sen MAD filter used to chop contiguous near/far tails
    (frame 84, pre-refinement). Windows short on points expand to the
    min_window_points nearest samples.

    Same input/return contract as estimate_pitch_from_widths.
    """
    depths, Y_3d = _preprocess_widths(widths, f_x, f_y, image_height, w_real,
                                      z_cap_m)
    z_range = float(depths[-1] - depths[0]) if len(depths) >= 2 else 0.0
    if z_range < min_valid_range_m or len(depths) < 2:
        return _empty_result()

    z_vis_min, z_vis_max = float(depths[0]), float(depths[-1])
    z_samps = np.linspace(z_vis_min, z_vis_max, n_pitch_samples)
    pitch_samps = np.empty(n_pitch_samples)
    y_samps = np.empty(n_pitch_samples)

    for i, zc in enumerate(z_samps):
        half = max(window_min_m, window_frac * zc)
        lo = np.searchsorted(depths, zc - half)
        hi = np.searchsorted(depths, zc + half, side="right")
        if hi - lo < min_window_points:
            k = min(min_window_points, len(depths))
            idx = np.argpartition(np.abs(depths - zc), k - 1)[:k]
            zw, yw = depths[idx], Y_3d[idx]
        else:
            zw, yw = depths[lo:hi], Y_3d[lo:hi]
        if np.ptp(zw) < 1e-9:
            # all window points at one depth — no slope information; carry
            # the previous sample (z_samps ascends, windows overlap)
            pitch_samps[i] = pitch_samps[i - 1] if i else 0.0
            y_samps[i] = float(yw.mean())
            continue
        fit = theilslopes(yw, zw)
        pitch_samps[i] = np.degrees(np.arctan(fit.slope))
        y_samps[i] = fit.intercept + fit.slope * zc

    def pitch_at(z):
        z_c = np.clip(z, z_vis_min, z_vis_max)
        return float(np.interp(z_c, z_samps, pitch_samps))

    return {
        "pitch_at": pitch_at,
        "z_samples": z_samps,
        "pitch_samples": pitch_samps,
        "y_samples": y_samps,
        "z_points": depths,
        "y_points": Y_3d,
        "z_visible_min": z_vis_min,
        "z_visible_max": z_vis_max,
    }


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
        (y_pixel, pixel_width) pairs from sample_widths_from_curves.
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
        y_samples     : fitted Y_3d curve evaluated at z_samples (spline or
                        Theil-Sen line) — for profile debugging
        z_points      : depths of the points that survived all filters and
                        were actually fitted
        y_points      : corresponding Y_3d values
        z_visible_min : float
        z_visible_max : float
    On degenerate/short input pitch_at is None and the sample arrays are empty.
    """
    depths, Y_3d = _preprocess_widths(widths, f_x, f_y, image_height, w_real,
                                      z_cap_m)

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
        return _empty_result()

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
            "y_samples": res.intercept + res.slope * z_samps,
            "z_points": depths,
            "y_points": Y_3d,
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
        "y_samples": spl(z_samps),
        "z_points": depths,
        "y_points": Y_3d,
        "z_visible_min": z_vis_min,
        "z_visible_max": z_vis_max,
    }
