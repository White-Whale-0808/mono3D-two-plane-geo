import numpy as np
from scipy.stats import theilslopes


def _widths_to_depth_height(widths, f_x, f_y, image_height, w_real):
    """Shared preprocessing: IQR filter -> (depths, Y_3d) sorted by depth."""
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        raise ValueError(
            "widths must be shape (N, 2) with N > 0, got {}".format(widths.shape))
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    valid = (w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)
    widths = widths[valid]
    if len(widths) == 0:
        raise ValueError("all samples removed by IQR filter")
    depths = f_x * w_real / widths[:, 1]
    center_y = image_height / 2
    Y_3d = -depths * (widths[:, 0] - center_y) / f_y
    order = np.argsort(depths)
    return depths[order], Y_3d[order]



def fit_two_plane_model(widths, f_x, f_y, image_height, w_real,
                        min_knee_margin: float = 0.15,
                        improvement_threshold: float = 0.10):
    """Fit Y(z) with a continuous two-segment (hinge) model.

    Physical model: the road is (at most) two planes joined at one knee.
        Y(z) = b0 + s_near * z                      for z <  z_knee
        Y(z) = Y(z_knee) + s_far * (z - z_knee)     for z >= z_knee

    Grid-search the knee over the interior depth range; each candidate is an
    ordinary least-squares fit on the basis [1, z, relu(z - k)].  If the best
    two-plane fit does not reduce SSE by at least `improvement_threshold`
    relative to a single line, fall back to one plane (z_knee = None).

    Returns
    -------
    dict with keys:
        pitch_near_deg : float   pitch of the plane under / just ahead of ego
        pitch_far_deg  : float   pitch of the plane beyond the knee
                                 (== pitch_near_deg when one plane)
        z_knee         : float | None   distance to the plane change (m)
        z_visible_min, z_visible_max : float  measured depth coverage
        pitch_at       : callable z -> pitch_deg  (query any distance;
                                 clamped to the visible range semantics)
    """
    z, Y = _widths_to_depth_height(widths, f_x, f_y, image_height, w_real)

    # single-plane baseline (robust)
    res = theilslopes(Y, z)
    s_single = res.slope
    sse_single = float(((Y - (res.intercept + s_single * z)) ** 2).sum())

    # Too few samples to support a 3-dof hinge fit: report one plane.
    if len(z) < 6:
        pitch = float(np.degrees(np.arctan(s_single)))
        return {
            "pitch_near_deg": pitch, "pitch_far_deg": pitch, "z_knee": None,
            "z_visible_min": float(z[0]), "z_visible_max": float(z[-1]),
            "pitch_at": lambda zq, p=pitch: p,
        }

    lo = z[0] + min_knee_margin * (z[-1] - z[0])
    hi = z[-1] - min_knee_margin * (z[-1] - z[0])
    best = None
    for k in np.linspace(lo, hi, 25):
        A = np.column_stack([np.ones_like(z), z, np.maximum(z - k, 0.0)])
        coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        sse = float(((Y - A @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k, coef)

    sse2, k, (b0, s_near, d_s) = best
    if sse_single - sse2 < improvement_threshold * sse_single:
        # no meaningful knee: one plane
        pitch = float(np.degrees(np.arctan(s_single)))
        return {
            "pitch_near_deg": pitch, "pitch_far_deg": pitch, "z_knee": None,
            "z_visible_min": float(z[0]), "z_visible_max": float(z[-1]),
            "pitch_at": lambda zq, p=pitch: p,
        }

    pitch_near = float(np.degrees(np.arctan(s_near)))
    pitch_far = float(np.degrees(np.arctan(s_near + d_s)))

    def pitch_at(zq):
        return pitch_near if zq < k else pitch_far

    return {
        "pitch_near_deg": pitch_near, "pitch_far_deg": pitch_far,
        "z_knee": float(k),
        "z_visible_min": float(z[0]), "z_visible_max": float(z[-1]),
        "pitch_at": pitch_at,
    }



def estimate_pitch_from_widths(widths, f_x, f_y, image_height, w_real,
                               num_depth_bands: int = 5,
                               min_band_extent_m: float = 1.5,
                               min_band_samples: int = 6,
                               min_profile_range_m: float = 3.0,
                               min_valid_range_m: float = 0.5,
                               z_cap_m: float = 45.0,
                               resid_mad_k: float = 5.0,
                               return_debug: bool = False):
    """Estimate per-depth pitch angle from per-band lane widths.

    Banding is adaptive: fitting a slope on a depth slice that is too thin or
    too sparse amplifies width noise into tens of degrees of pitch error
    (observed empirically: frames with z-range < 4 m had ~6x the profile MAE
    of frames with z-range >= 4 m). The number of bands is therefore capped so
    each band spans at least ``min_band_extent_m`` of depth and holds at least
    ``min_band_samples`` samples; when the whole visible range is shorter than
    ``min_profile_range_m`` a single global fit is returned, and when it is
    shorter than ``min_valid_range_m`` no estimate is returned at all.

    Parameters
    ----------
    widths : np.ndarray of shape (N, 2)
        (y_pixel, pixel_width) pairs sampled along the visible road.
    f_x, f_y : float
        Camera focal lengths (pixels) for the resized image.
    image_height : int
        Image height in pixels (for principal-point computation).
    w_real : float
        Real-world lane width (m).
    num_depth_bands : int
        Upper bound on the number of depth intervals. Each produces one
        (z_center, pitch_deg) estimate. Default 5.
    min_band_extent_m : float
        Minimum depth (m) a band must span. Default 1.5.
    min_band_samples : int
        Minimum samples per band. Default 6.
    min_profile_range_m : float
        Visible z-range below which a single global Theil-Sen fit replaces
        the per-band profile. Default 3.0.
    min_valid_range_m : float
        Visible z-range below which no estimate is attempted (returns []).
        Default 0.5.
    z_cap_m : float
        Samples deeper than this are dropped: depth scales as 1/pixel_width,
        so a few-pixel-wide spurious detection near the vanishing point maps
        to a huge depth and would dominate the band edges. Default 45.
    resid_mad_k : float
        Outlier gate on residuals from a global Theil-Sen fit, in robust
        sigmas (1.4826*MAD). Removes off-plane junk (kerbs, crosswalks,
        mis-associated markings) that the width-IQR filter cannot see.
        Default 5.0.
    return_debug : bool
        If True, return ``(pitch_per_depth, depths, Y_3d)`` instead of just
        ``pitch_per_depth``.

    Returns
    -------
    pitch_per_depth : list of (float, float)
        [(z_center_m, pitch_deg), ...] sorted by increasing depth.
        Empty when depth coverage is too short to support any estimate.
    """
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        raise ValueError(
            "estimate_pitch_from_widths: widths must be shape (N, 2) with N > 0, "
            "got shape {}. Left/right lane fits likely have no overlapping y-range "
            "or produced no valid sample points.".format(widths.shape)
        )

    # IQR outlier filter on pixel width
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    valid = (w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)
    widths = widths[valid]

    if len(widths) == 0:
        raise ValueError(
            "estimate_pitch_from_widths: all {} sample(s) were removed by the "
            "IQR filter — lane widths too inconsistent to estimate pitch.".format(len(w))
        )

    depths = f_x * w_real / widths[:, 1]
    center_y = image_height / 2
    Y_3d = -depths * (widths[:, 0] - center_y) / f_y

    # Depth cap: z ~ 1/pixel_width, so spurious few-pixel widths explode to
    # unphysical depths and would stretch the band edges over junk.
    in_range = depths <= z_cap_m
    if in_range.sum() >= 2:
        depths, Y_3d = depths[in_range], Y_3d[in_range]

    # Sort by depth
    sort_idx = np.argsort(depths)
    depths = depths[sort_idx]
    Y_3d = Y_3d[sort_idx]

    # Robust residual filter: global Theil-Sen fit, drop samples further than
    # resid_mad_k robust sigmas. Width-IQR can't catch off-plane junk whose
    # width happens to be plausible; vertical (Y) residuals can.
    if len(depths) >= 4:
        fit = theilslopes(Y_3d, depths)
        resid = Y_3d - (fit.intercept + fit.slope * depths)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 1e-9:
            keep = np.abs(resid - med) <= resid_mad_k * 1.4826 * mad
            if keep.sum() >= 2:
                depths, Y_3d = depths[keep], Y_3d[keep]

    # Adaptive band count: every band must span >= min_band_extent_m of depth
    # and hold >= min_band_samples samples; degenerate coverage -> 1 band or
    # no estimate.
    z_range = float(depths[-1] - depths[0])
    if z_range < min_valid_range_m or len(depths) < 2:
        # Depth coverage too short for any slope to be meaningful.
        if return_debug:
            return [], depths, Y_3d
        return []
    if z_range < min_profile_range_m:
        n_bands = 1
    else:
        n_bands = max(1, min(num_depth_bands,
                             int(z_range / min_band_extent_m),
                             len(depths) // min_band_samples))

    # Per-depth-band Theil-Sen regression → (z_center, pitch_deg)
    band_edges = np.linspace(depths.min(), depths.max(), n_bands + 1)
    pitch_per_depth = []

    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i + 1]
        mask = (depths >= lo) & (depths <= hi)
        if mask.sum() < 2:
            continue
        result = theilslopes(Y_3d[mask], depths[mask])
        pitch_rad = np.arctan(result.slope)
        pitch_deg = float(np.degrees(pitch_rad))
        z_center = float((lo + hi) / 2)
        pitch_per_depth.append((z_center, pitch_deg))

    if return_debug:
        return pitch_per_depth, depths, Y_3d
    return pitch_per_depth
