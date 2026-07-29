import numpy as np

# Segment shadowing: a segment consistently more OUTER than a parallel kept
# segment (by > margin px over >= min-overlap rows) is an outer paint edge or
# the outer line of a double marking. It never defines the inner envelope —
# where the inner-line segment is absent (dashed gaps, image bottom) the
# envelope must yield a GAP, not fall back outward: mid-chain fallback steps
# in x_inner(y) become width steps and blow up the spline pitch at the ends
# (frames 145-152, +1.3 to +2.1 deg MAE).
_SHADOW_MARGIN_PX = 3.0
_SHADOW_MIN_OVERLAP_ROWS = 8

# Fragment split: a 1-row x step in the envelope beyond the steepest
# legitimate lane slope (min_slope 0.3 -> <= ~3.3 px/row) means the envelope
# switched to a different segment, not lane curvature — fragment boundary.
_FRAG_MAX_STEP_PX = 4.0

# Junction consistency: bridging from one fragment to the next must stay
# within this off-direction displacement (bridge slope vs either fragment's
# own end slope, times the gap dy). Same 20 px scale validated for the old
# point-pair guards (12-frame scan 2026-07-16): the bottom outer-edge
# fallback lands 27-93 px outward, legitimate bridges stay under 20 px.
_JUNCTION_TOL_PX = 20.0

# End slopes near a junction use up to this many row pairs of the fragment.
_JUNCTION_SLOPE_ROWS = 10

# Edge refinement: search window (px) around the ELSED-derived envelope x.
# Small on purpose — the prior is already on the edge, and on a double
# marking the far-range gap between the two stripes shrinks to a few px;
# a wider window would let the OUTER stripe's flank (same gradient sign)
# capture the point.
_REFINE_SEARCH_PX = 3

# Minimum |dI/dx| (gray levels / px, 3-row averaged) to accept a refined
# edge. Paint-road contrast in CARLA is ~25+; below this there is no clear
# edge under the window (mask boundary, washed-out far range) and the
# ELSED prior is kept unchanged.
_REFINE_MIN_GRAD = 4.0


def inner_chain_points(segments, is_left, return_debug=False):
    """Clean inner-lane-line points for one side, from tracked segments.

    w_real (3.216 m) is the INNER-edge-to-inner-edge lane width, but the lane
    tracker deliberately keeps the whole marking group (all parallel paint
    edges — evidence for tracking). This function recovers the measurement
    semantics downstream:

    1. Per-row inner envelope (no tunable): for every image row y covered by
       at least one segment, x_inner(y) = innermost x among the covering
       segments (max x on the left side, min x on the right). Rows with no
       coverage stay gaps — nothing is fabricated, and density is NOT shaped
       here: the lane curve is a continuous model and pitch_estimation
       resamples widths z-uniformly on its own.
    2. Fragments: the envelope splits at coverage gaps and at 1-row x steps
       beyond _FRAG_MAX_STEP_PX (envelope switched segments).
    3. Junction consistency: a bridge between adjacent fragments whose
       direction deviates from both fragments' own end directions by more
       than _JUNCTION_TOL_PX (scaled by the gap dy) marks a break; the
       largest consistent fragment group (by row count) survives. One rule
       drops both mid-chain off-lane runs (artifact next to a dashed gap,
       frames 145-156) and bottom outer-edge fallback rows (inner segment
       ending above the image bottom, 27-93 px outward).

    Returns an (N, 2) array of (x, y) points, near (large y) first.
    With return_debug=True returns (points, dbg) where dbg exposes the
    intermediates for step visualization: shadowed / kept segments, envelope
    rows, fragments, junction groups and the index of the surviving group.
    """
    def _ret(points, dbg):
        return (points, dbg) if return_debug else points

    dbg = {"shadowed": np.empty((0, 4)), "kept_segments": np.empty((0, 4)),
           "rows": [], "fragments": [], "groups": [], "best_group": -1}
    segs = np.asarray(segments, dtype=np.float64)
    if segs.size == 0:
        return _ret(np.empty((0, 2)), dbg)
    y_lo = np.minimum(segs[:, 1], segs[:, 3])
    y_hi = np.maximum(segs[:, 1], segs[:, 3])

    def x_at(seg, y):
        x1, sy1, x2, sy2 = seg
        return (x1 + x2) / 2 if sy2 == sy1 else x1 + (y - sy1) * (x2 - x1) / (sy2 - sy1)

    # 0. shadowing: exclude outer paint edges / outer double-marking lines so
    # the envelope never falls back outward where the inner line is absent
    inner_sign = 1.0 if is_left else -1.0
    keep = np.ones(len(segs), dtype=bool)
    for i in range(len(segs)):
        for j in range(len(segs)):
            if i == j:
                continue
            lo = max(y_lo[i], y_lo[j])
            hi = min(y_hi[i], y_hi[j])
            if hi - lo < _SHADOW_MIN_OVERLAP_ROWS:
                continue
            d_lo = (x_at(segs[j], lo) - x_at(segs[i], lo)) * inner_sign
            d_hi = (x_at(segs[j], hi) - x_at(segs[i], hi)) * inner_sign
            if d_lo > _SHADOW_MARGIN_PX and d_hi > _SHADOW_MARGIN_PX:
                keep[i] = False
                break
    if keep.any():
        dbg["shadowed"] = segs[~keep]
        segs, y_lo, y_hi = segs[keep], y_lo[keep], y_hi[keep]
    dbg["kept_segments"] = segs

    # 1. dense per-row inner envelope, bottom (near) row first
    rows = []  # (y, x_inner), y descending
    for y in range(int(np.floor(y_hi.max())), int(np.ceil(y_lo.min())) - 1, -1):
        xs = []
        for (x1, y1, x2, y2), lo, hi in zip(segs, y_lo, y_hi):
            if not (lo <= y <= hi):
                continue
            xs.append((x1 + x2) / 2 if y1 == y2
                      else x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if xs:
            rows.append((float(y), max(xs) if is_left else min(xs)))
    dbg["rows"] = rows
    if not rows:
        return _ret(np.empty((0, 2)), dbg)

    # 2. fragments: contiguous runs with per-row x steps within the slope gate
    frags = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        if prev[0] - cur[0] == 1 and abs(cur[1] - prev[1]) <= _FRAG_MAX_STEP_PX:
            frags[-1].append(cur)
        else:
            frags.append([cur])

    # 3. junction consistency -> keep the largest consistent fragment group
    def _end_slope(frag, head):
        """Median dx-per-row near one fragment end; None for a single row."""
        pairs = list(zip(frag, frag[1:]))
        if not pairs:
            return None
        pairs = pairs[:_JUNCTION_SLOPE_ROWS] if head else pairs[-_JUNCTION_SLOPE_ROWS:]
        slopes = sorted((b[1] - a[1]) / (a[0] - b[0]) for a, b in pairs)
        return slopes[len(slopes) // 2]

    all_slopes = sorted((b[1] - a[1]) / (a[0] - b[0])
                        for f in frags for a, b in zip(f, f[1:]))
    global_med = all_slopes[len(all_slopes) // 2] if all_slopes else 0.0

    groups = [[frags[0]]]
    for prev_frag, frag in zip(frags, frags[1:]):
        dy = prev_frag[-1][0] - frag[0][0]  # >= 1
        bridge = (frag[0][1] - prev_frag[-1][1]) / dy
        refs = [s for s in (_end_slope(prev_frag, head=False),
                            _end_slope(frag, head=True)) if s is not None]
        if not refs:
            refs = [global_med]
        # consistent with EITHER side is enough: across a two-plane hinge the
        # bridge matches the far side long before it matches the near side
        dev = min(abs(bridge - r) for r in refs) * dy
        if dev > _JUNCTION_TOL_PX:
            groups.append([])
        groups[-1].append(frag)
    best = max(groups, key=lambda g: sum(len(f) for f in g))
    dbg["fragments"] = frags
    dbg["groups"] = groups
    dbg["best_group"] = groups.index(best)

    return _ret(np.array([(x, y) for f in best for y, x in f]), dbg)

def refine_inner_points(image_rgb, points, is_left):
    """Sub-pixel edge refinement of inner-chain points on the ORIGINAL image.

    inner_chain_points is limited by ELSED's straight-segment approximation
    (its information ceiling is the segment endpoints). This snaps each point
    to the actual paint→road transition: sample a 3-row-averaged horizontal
    gradient profile around the prior x, take the strongest gradient of the
    expected sign (inner edge of the LEFT marking is bright→dark going right,
    of the RIGHT marking dark→bright), and localize it to sub-pixel with a
    parabola fit over the peak's 3 gradient samples.

    Uses the unmasked image — the road mask boundary creates fake gradients
    and can cut through markings. Points with no clear edge in the window
    (|dI/dx| < _REFINE_MIN_GRAD) or too close to the border keep their prior.

    Parameters
    ----------
    image_rgb : PIL.Image or ndarray (H, W, 3), RGB
    points : (N, 2) ndarray of (x, y) from inner_chain_points
    is_left : bool

    Returns a new (N, 2) ndarray; y values are unchanged.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return pts
    gray = np.asarray(image_rgb, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    height, width = gray.shape

    sign = -1.0 if is_left else 1.0  # expected dI/dx at the inner edge
    s = _REFINE_SEARCH_PX
    refined = pts.copy()
    for i, (x, y) in enumerate(pts):
        yi, xi = int(round(y)), int(round(x))
        if not (1 <= yi <= height - 2 and s + 1 <= xi <= width - s - 2):
            continue
        strip = gray[yi - 1:yi + 2, xi - s - 1:xi + s + 2].mean(axis=0)
        grad = sign * 0.5 * (strip[2:] - strip[:-2])  # grad[m] ↔ column xi-s+m
        # nearest qualifying local max, not the strongest: the prior is
        # already on the edge, and a stronger same-sign edge in the window
        # (e.g. the other stripe of a double marking at far range) must not
        # capture the point
        k, best = None, None
        for m in range(len(grad)):
            if grad[m] < _REFINE_MIN_GRAD:
                continue
            if m > 0 and grad[m] < grad[m - 1]:
                continue
            if m < len(grad) - 1 and grad[m] < grad[m + 1]:
                continue
            key = (abs(m - s), -grad[m])
            if best is None or key < best:
                best, k = key, m
        if k is None:
            continue
        delta = 0.0
        if 0 < k < len(grad) - 1:
            denom = grad[k - 1] - 2 * grad[k] + grad[k + 1]
            if abs(denom) > 1e-9:
                delta = float(np.clip(0.5 * (grad[k - 1] - grad[k + 1]) / denom,
                                      -0.5, 0.5))
        refined[i, 0] = xi - s + k + delta
    return refined


def lane_curve(points):
    """Continuous image-space lane-line curve from inner-chain points.

    The polyline through the chain points IS the lane-line model: chain gaps
    (dashed markings, shadow-excluded rows) are bridged linearly between the
    neighbouring points, and kinks (two-plane hinge) are preserved exactly.
    Smoothing is deliberately NOT done here — the pitch spline downstream
    owns smoothing, with its physically-motivated 1/z² weights.

    Returns {"y": ascending ndarray, "x": ndarray}, or None with < 2 points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return None
    order = np.argsort(pts[:, 1])
    return {"y": pts[order, 1], "x": pts[order, 0]}


def curve_x_at(curve, ys):
    """x of the lane curve at ys (scalar or array). Callers must stay inside
    [curve.y[0], curve.y[-1]] — np.interp clamps, it never extrapolates."""
    return np.interp(ys, curve["y"], curve["x"])
