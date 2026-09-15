import numpy as np

"""
Lane segmentation via near-to-far continuity tracking, with geometry-derived
thresholds.

Tracking design (see docs/papers/lane_segmentation_design_logic.drawio):
  seed at the bottom (innermost rule), track upward band by band using a local
  line model, re-seed past dead ends (junctions).

Parameter derivation (see docs/papers/lane_segmentation_design_logic.drawio):
  Under the flat-ground pinhole model with camera height h,

      x - cx = f_x * X / z          z(y) = f_y * h / (y - cy)
      =>  x - cx = (f_x / f_y) * (X / h) * (y - cy)

  so every "lateral distance" threshold has an exact pixel form at row y:
      px(X, y) = f_x * X / z(y)
  and a lane line at lateral offset X has image slope dy/dx = f_y*h/(f_x*X).

  This lets us derive, instead of hand-tuning:
    - association tolerance  ~ a lateral distance to the side (_TOL_X_M)
    - cross-lane safety cap  ~ the lateral distance association may not cross
    - seed x-window          ~ the lateral band the ego's own marking lies in
    - noise slope gate       ~ |dy/dx| of a line 16 m to the side
    - model memory / reset   ~ expressed in metres via z(y)

  Note what is NOT needed: the lane WIDTH. Every threshold above is a lateral
  distance, and w_real only ever served to convert "a fraction of a lane" into
  metres. Stating the metres directly is what lets this stage run on a road of
  unknown width.

  z(y) is a flat-ground approximation; on the second plane it drifts, so
  (y - cy) is clamped and band-count fallbacks are kept as safety nets.

Camera geometry is REQUIRED. A hand-tuned fallback for un-calibrated cameras
used to live alongside this (min_slope / lane_band_tolerance / roi_near), but
every caller supplied the full calibration, so it was a second implementation
that nothing exercised and no test covered. Its thresholds were tuned to one
dataset in any case, so a genuinely new camera would need them re-derived, not
reused — which is what the geometry path does automatically.
"""

# Geometry constants (metres / fractions with physical meaning)
# Every threshold below is a LATERAL DISTANCE IN METRES, projected to pixels
# at each row by geom.px_at / px_max_at. They used to be written as fractions
# of w_real, which made the tracker look like it needed to know the lane width
# -- it does not: what it needs is how far to the side to look, and that is a
# property of driving, not of this road's markings. Measured over three routes
# (254 frames, assumed width swept 2.60-4.40 m): >=95% of frames produce
# BIT-IDENTICAL lane curves and the lateral positions never move at all, so a
# fixed distance costs nothing that a per-road width would buy. Values here are
# the exact equivalents of the old w_real=3.25 forms; see the git history for
# the re-derivation from road geometry.
_TOL_X_M             = 0.325  # association tol as a lateral distance
_TOL_PX_FLOOR        = 3.0    # ELSED endpoint noise floor (px)
_CROSS_X_M           = 1.30   # never search this far to the side: the cap that
                              # stops association crossing into the next lane
_SEED_X_LO_M         = 0.0    # seed window, near edge (ego may straddle the line)
_SEED_X_HI_M         = 3.25   # seed window, far edge
_SLOPE_GATE_X_M      = 3.25   # noise slope gate: |dy/dx| of a line this far aside
_ROI_X_M             = 3.25   # segment mid_x must lie within this of the axis
_SEED_X_MAX          = 8.0    # seed slope gate: lines beyond 8 m lateral are noise
_NOISE_X_MAX         = 16.0   # prefilter slope gate: beyond 16 m lateral
_MODEL_MEMORY_M      = 4.0    # local model fits points within 4 m of depth
_RESET_GAP_M         = 2.0    # detection gap > 2 m => possible plane change
_SUPPORT_MIN_LEN_PX  = 60.0   # lone seed-only re-seeded segments shorter than this
                              # are isolated blobs (poles/hillside: 32-54 px;
                              # legit single-seg far sections: >= 77 px)

# Camera model (z_at / z_min / lane_px bounds, grade-uncertainty constants)
# lives in geometry.py — paint_evidence uses the same bounds. `_Geometry` is
# the historical local name, kept so existing call sites and debug tooling
# stay valid.
from libs.inference.geometry import CameraGeometry as _Geometry


def _segment_info(segments, geom):
    """Precompute per-segment geometry; drop only clearly-horizontal noise."""
    # slope gate: dy/dx of a line _SLOPE_GATE_X_M to the side (a lane line at
    # lateral X has image slope f_y*h/(f_x*X)). Anything flatter cannot be a
    # lane line under normal driving — filters stop lines and crosswalks,
    # which have dy/dx ≈ 0.
    slope_gate = geom.f_y * geom.h / (geom.f_x * _SLOPE_GATE_X_M)

    infos = []
    for seg in np.asarray(segments, dtype=np.float64):
        x1, y1, x2, y2 = seg
        if x2 == x1:
            slope = np.inf  # vertical: perfectly valid lane candidate
        else:
            slope = (y2 - y1) / (x2 - x1)
        mid_y = (y1 + y2) / 2

        mid_x = (x1 + x2) / 2

        if np.isfinite(slope) and abs(slope) < slope_gate:
            continue

        # ROI filter: segment mid_x must be within _ROI_X_M of the image
        # centre. Uses the uphill worst-case bound so valid lane lines are
        # never excluded on ascending sections.
        # TODO: re-derive _ROI_X_M from the p95 lateral offset in CARLA GT.
        if abs(mid_x - geom.cx) > geom.px_max_at(_ROI_X_M, mid_y):
            continue

        infos.append({
            "seg": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            "p1": (x1, y1), "p2": (x2, y2),
            "slope": slope,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "y_min": min(y1, y2),
            "y_max": max(y1, y2),
        })
    return infos


def _x_on_segment_line(info, y):
    """x of the segment's own infinite line at height y."""
    x1, y1 = info["p1"]
    if not np.isfinite(info["slope"]) or info["slope"] == 0:
        return info["mid_x"]
    return x1 + (y - y1) / info["slope"]


def _fit_x_of_y(points, geom, y_now, last_n=8):
    """Fit x = a*y + b on the lane's recent direction.

    Normally the model spans points within _MODEL_MEMORY_M metres of the
    current row's depth — one road section, not the whole lane. Above the
    flat-ground validity limit z(y) saturates and cannot select a depth
    window, so the most recent `last_n` endpoints stand in.
    """
    pts = None
    if y_now is not None and geom.z_valid(y_now):
        z_now = geom.z_at(y_now)
        recent = [p for p in points
                  if geom.z_valid(p[1]) and z_now - geom.z_at(p[1]) <= _MODEL_MEMORY_M]
        if len(recent) >= 4:
            pts = recent
    if pts is None:
        pts = points[-last_n:]

    ys = np.array([p[1] for p in pts])
    xs = np.array([p[0] for p in pts])
    if len(pts) < 2 or np.ptp(ys) < 1e-6:
        return None
    a, b = np.polyfit(ys, xs, 1)
    return a, b


def _find_seed(infos, selected, is_left, center_x,
               band_edges, search_from_band, geom, first_seed=True):
    """Lowest band (index <= search_from_band) with an innermost seed group."""
    sign = -1.0 if is_left else 1.0

    # slope gate: a lane-parallel line within X m lateral distance.
    # The first seed sits on the ego plane where the flat model holds
    # (X_max = 8 m); re-seeds may land on the second plane whose lines
    # are flatter, so only the basic noise gate (16 m) is applied.
    x_gate = _SEED_X_MAX if first_seed else _NOISE_X_MAX
    seed_slope_gate = geom.f_y * geom.h / (geom.f_x * x_gate)

    def is_seed_candidate(info):
        s = info["slope"]
        if not np.isfinite(s):
            return True  # vertical is fine on either side near the camera
        on_side = info["mid_x"] < center_x if is_left else info["mid_x"] > center_x
        if not (on_side and s * sign > 0):
            return False
        if abs(s) < seed_slope_gate:
            return False
        # NOTE on a tempting-but-rejected idea: gating re-seeds by direction
        # consistency with the dying track (to block kerb / stop lines at
        # intersections, e.g. frame 448) was tried with both additive and
        # ratio tolerances — any setting tight enough to block the junk also
        # rejected legitimate re-seeds at crest plane changes (60+ frames
        # regressed). Re-seed direction alone does not separate the two.
        return True

    for i in range(search_from_band, -1, -1):
        lo, hi = band_edges[i], band_edges[i + 1]
        y_c = (lo + hi) / 2

        # Above the flat-ground validity limit the seed x-window is derived
        # from a collapsed z_min and spans half the image; poles and hillside
        # edges get seeded there. Never SEED in that region (association may
        # still track into it from below).
        if not geom.z_valid(y_c):
            continue

        # Ego sits somewhere inside its lane, so the inner marking lies at a
        # lateral X anywhere in [_SEED_X_LO_M, _SEED_X_HI_M]. Evaluated with
        # the uphill worst-case bound so the window stays wide on slopes.
        inner = geom.px_max_at(_SEED_X_LO_M, y_c)
        outer = geom.px_max_at(_SEED_X_HI_M, y_c)
        if is_left:
            x_lo, x_hi = center_x - outer, center_x - inner
        else:
            x_lo, x_hi = center_x + inner, center_x + outer
        group_tol = max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y_c))

        cands = []
        for info in infos:
            if info["seg"] in selected:
                continue
            if not (info["y_max"] >= lo and info["y_min"] <= hi):
                continue
            if not is_seed_candidate(info):
                continue
            x_c = _x_on_segment_line(info, y_c)
            if not (x_lo <= x_c <= x_hi):
                continue
            cands.append((x_c, info))
        if cands:
            # innermost + tolerance, same spirit as the old per-band rule
            if is_left:
                best = max(c[0] for c in cands)
                picked = [inf for x, inf in cands if x >= best - group_tol]
            else:
                best = min(c[0] for c in cands)
                picked = [inf for x, inf in cands if x <= best + group_tol]
            return picked, i
    return None, -1


def _track_side(infos, is_left, center_x, track_bands, geom):
    """Seed at the bottom, track upward by continuity; re-seed past dead ends."""
    sign = -1.0 if is_left else 1.0
    y_lo = min(i["y_min"] for i in infos)
    y_hi = max(i["y_max"] for i in infos)
    if y_hi - y_lo < 1:
        return []
    band_edges = np.linspace(y_lo, y_hi, track_bands + 1)

    def band_overlap(info, lo, hi):
        return info["y_max"] >= lo and info["y_min"] <= hi

    def assoc_window(y, missed):
        """Search half-width around the prediction at row y.

        The uphill worst-case bound keeps the window wide enough on
        ascending rows; the cross-lane cap uses the flat-ground scale,
        preserving the 4x headroom between the two.
        """
        # tolerance scale: worst-case uphill (largest pixel scale there);
        # cross-lane cap: worst-case flat (other lane closest in pixels).
        base = max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y))
        cap = max(_TOL_PX_FLOOR, geom.px_at(_CROSS_X_M, y))
        return min(cap, base * (1.0 + missed))

    def group_tol(y):
        """Same-physical-marking grouping width."""
        return max(_TOL_PX_FLOOR, geom.px_max_at(_TOL_X_M, y))

    selected = {}
    sections = []  # per seed: {"segs": [...], "first": bool, "extra_bands": int}
    search_from_band = track_bands - 1
    first_seed = True

    while search_from_band >= 0:
        seed_items, seed_band = _find_seed(
            infos, selected, is_left, center_x,
            band_edges, search_from_band, geom, first_seed)
        if seed_items is None:
            break
        section = {"segs": [], "first": first_seed, "extra_bands": 0}
        sections.append(section)
        first_seed = False

        track_points = []  # accepted (x, y) endpoints, bottom -> top
        for info in seed_items:
            selected[info["seg"]] = True
            section["segs"].append(info["seg"])
            for p in (info["p1"], info["p2"]):
                track_points.append(p)
        track_points.sort(key=lambda p: -p[1])  # by y descending (near first)
        last_accept_y = max(p[1] for p in track_points)

        # --- track upward from the seed band ---
        missed = 0
        stop_band = -1  # band index where this track gave up (-1: reached top)
        for i in range(seed_band - 1, -1, -1):
            lo, hi = band_edges[i], band_edges[i + 1]
            y_c = (lo + hi) / 2

            model = _fit_x_of_y(track_points, geom, y_c)
            if model is None:
                stop_band = i
                break
            a, b = model
            x_pred = a * y_c + b
            tol = assoc_window(y_c, missed)

            accepted = []
            for info in infos:
                if info["seg"] in selected or not band_overlap(info, lo, hi):
                    continue
                # a lane line on this side can never have the opposite slope
                # sign (seeds already enforce this; kerb and pole segments
                # were slipping in through association)
                if np.isfinite(info["slope"]) and info["slope"] * sign <= 0:
                    continue
                x_c = _x_on_segment_line(info, y_c)
                if abs(x_c - x_pred) > tol:
                    continue
                # never cross the centre line
                if is_left and x_c > center_x:
                    continue
                if not is_left and x_c < center_x:
                    continue
                accepted.append((abs(x_c - x_pred), x_c, info))

            if not accepted:
                missed += 1
                if missed > max(4, track_bands // 3):
                    stop_band = i
                    break
                continue

            # Plane-change reset: after a real gap the lane has likely kinked;
            # old points would drag the model toward the previous plane.
            # (above the flat-ground limit z(y) saturates and no depth gap can
            # be computed — fall back to counting missed bands)
            if geom.z_valid(y_c) and geom.z_valid(last_accept_y):
                gap_m = geom.z_at(y_c) - geom.z_at(last_accept_y)
                do_reset = gap_m > _RESET_GAP_M
            else:
                do_reset = missed >= 2
            if do_reset and missed > 0:
                track_points = track_points[-2:]
            missed = 0

            # keep everything close to the best match (parallel double markings
            # within the grouping width are the same physical lane)
            accepted.sort(key=lambda t: t[0])
            best_x = accepted[0][1]
            gtol = group_tol(y_c)
            for _, x_c, info in accepted:
                if abs(x_c - best_x) <= gtol:
                    selected[info["seg"]] = True
                    section["segs"].append(info["seg"])
                    for p in (info["p1"], info["p2"]):
                        track_points.append(p)
            section["extra_bands"] += 1
            last_accept_y = y_c

        if stop_band < 0:
            break  # reached the top of the road area
        # Re-seed strictly above the failure point for the next road section.
        search_from_band = stop_band - 1

    # A re-seeded section whose track never accepted a band beyond its own
    # seed group AND that is a lone short segment is an isolated blob (pole,
    # hillside edge, mask hole), not a road section. Multi-segment or long
    # seed-only sections are legitimate far lane sections that fit entirely
    # inside one seed group.
    kept = []
    for sec in sections:
        if sec["first"] or sec["extra_bands"] > 0 or len(sec["segs"]) >= 2:
            kept.extend(sec["segs"])
            continue
        x1, y1, x2, y2 = sec["segs"][0]
        if np.hypot(x2 - x1, y2 - y1) >= _SUPPORT_MIN_LEN_PX:
            kept.extend(sec["segs"])
    return kept


def split_left_right_lines(
    segments,
    image_width: int,
    img_height: int,
    track_bands: int = 16,
    *,
    f_x: float,
    f_y: float,
    camera_height: float,
):
    """Split ELSED segments into inner left / right lane segments.

    Parameters
    ----------
    segments : array-like, shape (N, 4)
        Raw ELSED output: each row is (x1, y1, x2, y2).
    image_width, img_height : int
    track_bands : int
        Tracking band count (internally clamped to >= 16). Independent of
        lane_fitting's num_bands — tracking steps and fitting knots are
        separate concepts.
    f_x, f_y, camera_height : float, keyword-only, REQUIRED
        Camera focal lengths (px) and camera height above road (m).
        Association tolerance, seed window, slope gates and model memory are
        all derived from these; there is no un-calibrated mode. The lane WIDTH
        is deliberately not taken: every threshold here is a lateral distance
        in metres, so the tracker works on a road whose width it does not know.

    Returns
    -------
    inner_left, inner_right : list of (x1, y1, x2, y2) tuples
    """
    segments = np.asarray(segments)
    if segments.size == 0:
        return [], []

    geom = _Geometry.without_lane_width(f_x, f_y, camera_height,
                                        image_width, img_height)

    infos = _segment_info(segments, geom)
    if not infos:
        return [], []

    center_x = image_width / 2
    track_bands = max(int(track_bands), 16)

    inner_left = _track_side(infos, True, center_x, track_bands, geom)
    inner_right = _track_side(infos, False, center_x, track_bands, geom)

    return inner_left, inner_right
