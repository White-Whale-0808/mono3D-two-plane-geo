import numpy as np
from scipy.stats import theilslopes
from scipy.interpolate import UnivariateSpline

# Windowed-estimator defaults, exposed for the step visualization: the local
# Theil-Sen window is |z - zc| <= max(WINDOW_MIN_M, WINDOW_FRAC * zc).
WINDOW_FRAC = 0.15
WINDOW_MIN_M = 1.0

# Near-field self-calibration: ground-plane depth window (m) and the θ0 gate.
# The window's lower bound clears the image margin, the upper bound limits the
# road-curvature error (relative error z²/(2·R·h) — sub-percent at 5 m for
# R ≥ 1 km). The gate rejects frames whose w(z) trend says the near field is
# not the support plane (grade transitions, and on a real car suspension
# transients). It admits at most w·θ0·z̄/h ≈ 0.055 m (1.7%) of width bias
# while sitting well above the θ0 noise floor on steady sections (full_road
# scan 2026-08-28: steady-section w std 0.2%, θ0 tracking dpitch/ds at
# corr 0.77).
#
# Known limitation on SUSTAINED grades (up/down_hile acceptance 2026-08-28):
# the vehicle body sits at a small direction-dependent pitch relative to the
# road (measured road_pitch − cam_pitch: −0.11° descending, +0.12° climbing),
# which pushes θ0 to ~±0.25-0.32° for whole sections — so the gate rejects
# 80-95% of frames there, and the ones it admits still carry a ±0.03-0.05 m
# residual. The principled fix is to CORRECT with the measured θ0 (the
# Theil-Sen intercept `w_real_z0` is already θ0-free) rather than only gate
# on it; that is the next step, not done here.
NEARFIELD_Z_MIN_M = 2.0
NEARFIELD_Z_MAX_M = 5.0
THETA0_GATE_DEG = 0.3
# Hold/adopt policy (full_road acceptance 2026-08-28, verified against the
# GT-projected implied width): a near-field measurement describes the road
# patch it was taken on, so a held value expires once the vehicle has driven
# past that patch — NEARFIELD_Z_MAX_M metres. And adoption requires the gate
# to stay open across a minimum stretch of travel: at a curvature INFLECTION
# the w(z) trend crosses zero while the bias is at its largest (θ0 is blind
# there for an isolated frame — s161 adopted a 2.9 reading that way), whereas
# genuine support-plane stretches pass for many metres in a row.
NEARFIELD_MAX_HOLD_M = NEARFIELD_Z_MAX_M
NEARFIELD_MIN_RUN_M = 0.5
# Frame-count fallbacks when the caller never feeds travel distance
# (assume ~0.1 m/frame — conservative for typical capture rates).
NEARFIELD_MAX_HOLD_FRAMES = 40
NEARFIELD_MIN_RUN_FRAMES = 4


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


def estimate_w_real_nearfield(widths, f_x, f_y, image_height, camera_height,
                              z_near_min=NEARFIELD_Z_MIN_M,
                              z_near_max=NEARFIELD_Z_MAX_M, min_points=8):
    """Per-frame w_real from the near-field ground plane (camera-height anchor).

    The near road is the plane the wheels sit on, and the camera is rigid on
    the vehicle, so in the camera frame that plane is at Y = -camera_height by
    construction — no flatness assumption about the road ahead. That gives a
    second, w_real-free depth equation valid only there:

        z_h = f_y·h / (y - cy)                     (ground-plane inverse perspective)

    and combining it with the pinhole width model w_px = f_x·w_real/z yields a
    per-row measurement of the physical lane width:

        w_real(y) = w_px · f_y·h / (f_x·(y - cy))

    Rows are kept where z_h ∈ [z_near_min, z_near_max]: the lower bound skips
    the hood/margin rows, the upper bound limits the road-curvature error
    (deviation from the support plane grows as z²/2R, i.e. relative error
    z²/(2·R·h) — sub-percent at 5 m for R ≥ 1 km).

    A Theil-Sen fit of w_real(y) against z_h separates the two calibration
    unknowns, because their error signatures differ: a wrong w_real shifts all
    rows equally, while an uncalibrated camera mounting pitch θ0 (positive =
    pitched down) scales z_h by (1 + θ0·z/h), a trend ∝ z. Hence the intercept
    at z=0 is the θ0-free width and the slope reads the mounting pitch:

        w_est(z) = w_real·(1 + θ0·z/h)  →  θ0 = slope·h/intercept

    Returns None when fewer than min_points rows land in the window (short
    curves, bottom occlusion); otherwise a dict:

        w_real_med : robust headline estimate (median over the window; carries
                     a +w·θ0·z̄/h bias if the mounting pitch is uncalibrated)
        w_real_z0  : Theil-Sen intercept — θ0-free, but an extrapolation to
                     z=0, so noisier than the median
        theta0_deg : implied camera mounting pitch (deg, positive = down)
        n_points   : rows used
        z_lo, z_hi : ground-plane depth range actually used (m)

    Input `widths` is the usual (y_pixel, pixel_width) array. The pipeline
    consumes this through NearfieldWidthCalibrator (θ0 gate + hold); the
    validation evidence lives in debug/w_real_nearfield_scan.py / _analysis.py.
    """
    widths = np.asarray(widths, dtype=float)
    if widths.ndim != 2 or widths.shape[1] != 2 or len(widths) == 0:
        return None
    y, w_px = widths[:, 0], widths[:, 1]
    dy = y - image_height / 2.0
    ok = (dy > 0) & (w_px > 0)          # below the horizon, physical width
    if not ok.any():
        return None
    z_h = f_y * camera_height / dy[ok]
    sel = (z_h >= z_near_min) & (z_h <= z_near_max)
    if sel.sum() < min_points:
        return None
    z = z_h[sel]
    w_est = z * w_px[ok][sel] / f_x
    if np.ptp(z) < 1e-9:
        return None
    fit = theilslopes(w_est, z)
    w_med = float(np.median(w_est))
    w_z0 = float(fit.intercept)
    theta0 = float(np.degrees(np.arctan(fit.slope * camera_height / w_z0))) \
        if w_z0 > 0 else np.nan
    return {
        "w_real_med": w_med,
        "w_real_z0": w_z0,
        "theta0_deg": theta0,
        "n_points": int(sel.sum()),
        "z_lo": float(z.min()),
        "z_hi": float(z.max()),
    }


def nearfield_widths_from_curves(left_curve, right_curve, f_y, camera_height,
                                 image_height, z_lo=NEARFIELD_Z_MIN_M,
                                 z_hi=NEARFIELD_Z_MAX_M):
    """Per-row (y, w_px) over the near-field window, straight off the curves.

    Every integer image row whose ground-plane depth f_y·h/(y−cy) falls in
    [z_lo, z_hi] and that both curves cover — denser than the z-uniform pitch
    samples, which is what the Theil-Sen θ0/width split wants.
    """
    if left_curve is None or right_curve is None:
        return np.empty((0, 2))
    cy = image_height / 2.0
    y_lo = max(left_curve["y"][0], right_curve["y"][0],
               cy + f_y * camera_height / z_hi)
    y_hi = min(left_curve["y"][-1], right_curve["y"][-1],
               cy + f_y * camera_height / z_lo)
    if not y_hi > y_lo:
        return np.empty((0, 2))
    rows = np.arange(np.ceil(y_lo), np.floor(y_hi) + 1.0)
    xl = np.interp(rows, left_curve["y"], left_curve["x"])
    xr = np.interp(rows, right_curve["y"], right_curve["x"])
    w = xr - xl
    keep = w > 0
    return np.column_stack([rows[keep], w[keep]])


class NearfieldWidthCalibrator:
    """Per-frame w_real from the near field: θ0-gated, bounded hold-last-valid.

    Wraps `estimate_w_real_nearfield` into the stateful policy the pipeline
    uses. A frame passes the gate when |θ0| <= THETA0_GATE_DEG (the near
    field IS the support plane); its width is adopted only once the gate has
    stayed open across NEARFIELD_MIN_RUN_M of travel (isolated passes at
    curvature inflections are the gate's blind spot). A rejected frame reuses
    the last adopted value while the vehicle is still within
    NEARFIELD_MAX_HOLD_M of the last accepted patch; beyond that the value
    is stale — width demonstrably changes across long rejected stretches
    (full_road road 41: 3.39 → 3.31) — and the configured w_real takes over
    as fallback.

    One instance per image sequence. Call `advance_to(dist_m)` with the
    cumulative travel distance before each frame to make the run/hold bounds
    metric; without it they fall back to frame counts. Stages 1–4 keep the
    configured w_real for their threshold derivations (insensitive at the
    few-percent level); only the metric stage consumes the calibrated value.
    """

    def __init__(self, f_x, f_y, image_height, camera_height, w_real_fallback,
                 theta0_gate_deg=THETA0_GATE_DEG):
        self.f_x = f_x
        self.f_y = f_y
        self.image_height = image_height
        self.camera_height = camera_height
        self.theta0_gate_deg = theta0_gate_deg
        self.w_real_fallback = float(w_real_fallback)
        self.w_real = float(w_real_fallback)   # value the last update() used
        self.last_estimate = None              # raw estimator dict, last frame
        self._dist = None                      # advance_to state (m)
        self._frame = -1
        self._run_start = None                 # (dist, frame) of gate-run start
        self._last_adopt = None                # (dist, frame) of last adoption
        self._held = None

    def advance_to(self, dist_m):
        """Cumulative travel distance (m) of the frame about to be fed."""
        self._dist = float(dist_m)

    def _span(self, since):
        """Travel since a (dist, frame) mark: metres if known, else frames."""
        if self._dist is not None and since[0] is not None:
            return self._dist - since[0], True
        return self._frame - since[1], False

    def update(self, left_curve, right_curve):
        """Feed one frame's lane curves; returns the w_real to use for it."""
        self._frame += 1
        widths = nearfield_widths_from_curves(
            left_curve, right_curve, self.f_y, self.camera_height,
            self.image_height)
        est = estimate_w_real_nearfield(
            widths, self.f_x, self.f_y, self.image_height, self.camera_height)
        self.last_estimate = est
        passed = est is not None and np.isfinite(est["theta0_deg"]) \
            and abs(est["theta0_deg"]) <= self.theta0_gate_deg
        if passed:
            if self._run_start is None:
                self._run_start = (self._dist, self._frame)
            span, metric = self._span(self._run_start)
            if span >= (NEARFIELD_MIN_RUN_M if metric
                        else NEARFIELD_MIN_RUN_FRAMES):
                self._held = est["w_real_med"]
                self._last_adopt = (self._dist, self._frame)
        else:
            self._run_start = None
        if self._held is not None:
            span, metric = self._span(self._last_adopt)
            if span <= (NEARFIELD_MAX_HOLD_M if metric
                        else NEARFIELD_MAX_HOLD_FRAMES):
                self.w_real = self._held
                return self.w_real
        self.w_real = self.w_real_fallback
        return self.w_real


def sample_widths_from_curves(left_curve, right_curve, num_samples, *,
                              f_x, w_real, samples_per_meter=None):
    """(y, width) samples over the y-overlap of two continuous lane curves.

    The curves come from lane_fitting.lane_curve: {"y": ascending, "x": ...},
    already gap-bridged. Width is a metric quantity (only meaningful through
    f_x / w_real), so sampling it belongs to this stage, not lane_fitting.

    Dense candidate sweep → depth via z = f_x·w_real/width → resample uniformly
    in z, so each metre of visible depth gets equal sample density. The count
    is `samples_per_meter` × the visible depth range, or a fixed `num_samples`
    when samples_per_meter is unset.

    Sampling uniformly in y instead was the pre-WWH-9 behaviour and is gone:
    y-uniform spends most of its samples on the near few metres (z ∝ 1/(y-cy)),
    which is exactly where the pitch profile needs them least.
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
