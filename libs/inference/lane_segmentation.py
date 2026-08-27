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
    - association tolerance  ~ fraction of the local lane pixel width
    - cross-lane safety cap  ~ 0.4 lane width in pixels at that row
    - seed x-window          ~ ego sits within its lane: X in (0.5±δ)·w_real
    - noise slope gate       ~ |dy/dx| of a line 16 m to the side
    - model memory / reset   ~ expressed in metres via z(y)

  z(y) is a flat-ground approximation; on the second plane it drifts, so
  (y - cy) is clamped and band-count fallbacks are kept as safety nets.
  When camera geometry is not provided, the legacy hand-tuned behaviour is
  used unchanged.
"""

# Geometry constants (metres / fractions with physical meaning)
_TOL_LANE_FRACTION   = 0.10   # association tol = 10% of local lane width
_TOL_PX_FLOOR        = 3.0    # ELSED endpoint noise floor (px)
_CROSS_LANE_FRACTION = 0.40   # never search beyond 40% of a lane width
_SEED_DELTA          = 0.5    # ego lateral position uncertainty (lane fractions)
                              # = max lateral offset / w_real = (w_real/2) / w_real
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


def _segment_info(segments, min_slope, img_height, geom):
    """Precompute per-segment geometry; drop only clearly-horizontal noise."""
    if geom is not None:
        # slope gate: dy/dx of a line at the inner lane edge with maximum
        # vehicle lateral offset (X = w_real/2 + 0.5*w_real = w_real).
        # Anything flatter cannot be a lane line under normal driving —
        # filters stop lines and crosswalks which have dy/dx ≈ 0.
        slope_gate = geom.f_y * geom.h / (geom.f_x * geom.w)

    infos = []
    for seg in np.asarray(segments, dtype=np.float64):
        x1, y1, x2, y2 = seg
        if x2 == x1:
            slope = np.inf  # vertical: perfectly valid lane candidate
        else:
            slope = (y2 - y1) / (x2 - x1)
        mid_y = (y1 + y2) / 2

        mid_x = (x1 + x2) / 2

        if np.isfinite(slope):
            if geom is not None:
                if abs(slope) < slope_gate:
                    continue
            else:
                # legacy fallback: relaxed depth-adaptive gate
                if abs(slope) < 0.5 * min_slope * (mid_y / img_height):
                    continue

        # ROI filter: segment mid_x must be within one lane width of image
        # centre. Uses lane_px_max (uphill worst-case) so valid lane lines are
        # never excluded on ascending sections.
        # TODO: replace 1.0 multiplier with p95 lateral offset from CARLA GT.
        if geom is not None:
            if abs(mid_x - geom.cx) > geom.lane_px_max(mid_y):
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


def _fit_x_of_y(points, geom=None, y_now=None, last_n=8):
    """Fit x = a*y + b on the lane's recent direction.

    With geometry: use points within _MODEL_MEMORY_M metres of the current
    row's depth (the model should span one road section, not the whole lane).
    Without: the most recent `last_n` endpoints.
    """
    pts = None
    if geom is not None and y_now is not None and geom.z_valid(y_now):
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


def _find_seed(infos, selected, is_left, center_x, base_tol, roi_near,
               band_edges, search_from_band, geom, first_seed=True):
    """Lowest band (index <= search_from_band) with an innermost seed group."""
    sign = -1.0 if is_left else 1.0

    if geom is not None:
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
        if geom is not None and abs(s) < seed_slope_gate:
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
        if geom is not None and not geom.z_valid(y_c):
            continue

        if geom is not None:
            # Ego sits somewhere inside its lane: the inner marking lies at
            # lateral X in (0.5±δ)·w_real. The pixel window is evaluated with
            # z_min (worst-case uphill) so it stays wide enough on slopes.
            inner = geom.f_x * geom.w * (0.5 - _SEED_DELTA) / geom.z_min(y_c)
            outer = geom.f_x * geom.w * (0.5 + _SEED_DELTA) / geom.z_min(y_c)
            if is_left:
                x_lo, x_hi = center_x - outer, center_x - inner
            else:
                x_lo, x_hi = center_x + inner, center_x + outer
            group_tol = max(_TOL_PX_FLOOR, _TOL_LANE_FRACTION * geom.lane_px_max(y_c))
        else:
            if is_left:
                x_lo, x_hi = center_x * roi_near, center_x
            else:
                x_lo, x_hi = center_x, center_x * (2 - roi_near)
            group_tol = base_tol

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


def _track_side(infos, is_left, center_x, img_height, base_tol, roi_near,
                track_bands, geom):
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

        lane_px_max keeps the window wide enough on uphill rows; the
        cross-lane cap uses the same scale, preserving the 4x headroom.
        """
        if geom is not None:
            # tolerance scale: worst-case uphill (largest pixel scale there);
            # cross-lane cap: worst-case flat (other lane closest in pixels).
            base = max(_TOL_PX_FLOOR, _TOL_LANE_FRACTION * geom.lane_px_max(y))
            cap = max(_TOL_PX_FLOOR, _CROSS_LANE_FRACTION * geom.lane_px(y))
            return min(cap, base * (1.0 + missed))
        tol = base_tol * (2.0 + 1.5 * missed)
        return min(tol, 0.18 * center_x)

    def group_tol(y):
        """Same-physical-marking grouping width."""
        if geom is not None:
            return max(_TOL_PX_FLOOR, _TOL_LANE_FRACTION * geom.lane_px_max(y))
        return base_tol

    selected = {}
    sections = []  # per seed: {"segs": [...], "first": bool, "extra_bands": int}
    search_from_band = track_bands - 1
    first_seed = True

    while search_from_band >= 0:
        seed_items, seed_band = _find_seed(
            infos, selected, is_left, center_x, base_tol, roi_near,
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
            if geom is not None and geom.z_valid(y_c) and geom.z_valid(last_accept_y):
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
    min_slope: float,
    img_height: int,
    lane_band_tolerance: float,
    roi_near: float = 0.3,
    track_bands: int = 16,
    *,
    f_x: float = None,
    f_y: float = None,
    camera_height: float = None,
    w_real: float = None,
):
    """Split ELSED segments into inner left / right lane segments.

    Parameters
    ----------
    segments : array-like, shape (N, 4)
        Raw ELSED output: each row is (x1, y1, x2, y2).
    image_width, img_height : int
    min_slope, lane_band_tolerance, roi_near : float
        Legacy hand-tuned thresholds — used ONLY when camera geometry is not
        provided (see below).
    track_bands : int
        Tracking band count (internally clamped to >= 16). Independent of
        lane_fitting's num_bands — tracking steps and fitting knots are
        separate concepts.
    f_x, f_y, camera_height, w_real : float, keyword-only
        Camera focal lengths (px), camera height above road (m) and real lane
        width (m). When ALL are given, association tolerance, seed window,
        slope gates and model memory are derived from projection geometry
        instead of the legacy hand-tuned values.

    Returns
    -------
    inner_left, inner_right : list of (x1, y1, x2, y2) tuples
    """
    segments = np.asarray(segments)
    if segments.size == 0:
        return [], []

    geom = None
    if all(v is not None for v in (f_x, f_y, camera_height, w_real)):
        geom = _Geometry(f_x, f_y, camera_height, w_real, image_width, img_height)

    infos = _segment_info(segments, min_slope, img_height, geom)
    if not infos:
        return [], []

    center_x = image_width / 2
    track_bands = max(int(track_bands), 16)

    inner_left = _track_side(
        infos, True, center_x, img_height, lane_band_tolerance, roi_near,
        track_bands, geom)
    inner_right = _track_side(
        infos, False, center_x, img_height, lane_band_tolerance, roi_near,
        track_bands, geom)

    return inner_left, inner_right
