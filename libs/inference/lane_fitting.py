import numpy as np
import cv2
# from sklearn.linear_model import RANSACRegressor

"""
Use points view to do piecewise linear fitting
"""
def collect_points_from_segments(segments, extra_points_per_segment):
    points = []

    for seg in segments:
        x1, y1, x2, y2 = seg
        """
        Add the extra points along the line segment to make the fitting more robust
        """
        for t in np.linspace(0, 1, extra_points_per_segment):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append((x, y))
    return np.array(points)  # Output shape: (N, 2), list of (x, y) points

def piecewise_linear_fit(points, num_bands):

    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    band_edges = np.linspace(y_min, y_max, num_bands + 1)

    fits = []
    for i in range(num_bands):

        y_lower_bound, y_upper_bound = band_edges[i], band_edges[i + 1]
        mask = (points[:, 1] >= y_lower_bound) & (points[:, 1] < y_upper_bound)
        band_points = points[mask]

        if len(band_points) < 2:
            continue

        # RANSAC was tried here but made results worse: far bands have few points with
        # natural perspective spread, so a fixed residual_threshold excludes too many
        # valid points and causes the fit to go off.
        coeffs = np.polyfit(band_points[:, 1], band_points[:, 0], deg=1)
        a, b = coeffs

        fits.append({
            "y_start": y_lower_bound,
            "y_end": y_upper_bound,
            "slope": a,
            "intercept": b,
            "num_points": len(band_points)
        })

    return fits
"""
get x-coordinate at a specific y band based on the piecewise linear fits
"""
def get_x_at_y(fits, y):
    for f in fits:
        if f["y_start"] <= y <= f["y_end"]:
            return f["slope"] * y + f["intercept"]
    return None

def compute_lane_widths(left_fits, right_fits, num_samples, *, f_x=None, w_real=None,
                        samples_per_meter=None):
    left_y_min = min(f["y_start"] for f in left_fits)
    left_y_max = max(f["y_end"] for f in left_fits)
    right_y_min = min(f["y_start"] for f in right_fits)
    right_y_max = max(f["y_end"] for f in right_fits)

    y_min = max(left_y_min, right_y_min)
    y_max = min(left_y_max, right_y_max)

    if f_x is not None and w_real is not None:
        # z-space uniform sampling: dense candidate sweep → convert to depth →
        # resample uniformly in depth so each 1 m band gets equal point density.
        # Width is interpolated directly (not re-evaluated via get_x_at_y) to
        # avoid returning None for y values that fall in piecewise-band gaps.
        candidate_ys = np.linspace(y_min, y_max, 2000)
        ws, ys_valid = [], []
        for y in candidate_ys:
            x_l = get_x_at_y(left_fits, y)
            x_r = get_x_at_y(right_fits, y)
            if x_l is not None and x_r is not None and x_r > x_l:
                ws.append(x_r - x_l)
                ys_valid.append(y)
        if len(ws) < 2:
            return np.empty((0, 2))
        ws = np.array(ws)
        ys_valid = np.array(ys_valid)
        zs = f_x * w_real / ws
        order = np.argsort(zs)
        zs_s, ys_s, ws_s = zs[order], ys_valid[order], ws[order]
        if samples_per_meter is not None:
            # Scene-invariant density: n scales with the visible depth range so
            # each meter gets the same number of samples regardless of how far
            # the lanes are visible. Capped at the candidate sweep resolution
            # (interpolation cannot add information beyond it).
            z_range = zs_s[-1] - zs_s[0]
            n_samples = int(np.clip(np.ceil(z_range * samples_per_meter),
                                    2, len(candidate_ys)))
        else:
            n_samples = num_samples
        target_zs = np.linspace(zs_s[0], zs_s[-1], n_samples)
        target_ys = np.interp(target_zs, zs_s, ys_s)
        target_ws = np.interp(target_zs, zs_s, ws_s)
        widths = list(zip(target_ys, target_ws))
    else:
        sample_ys = np.linspace(y_min, y_max, num_samples)
        widths = []
        for y in sample_ys:
            x_left = get_x_at_y(left_fits, y)
            x_right = get_x_at_y(right_fits, y)
            if x_left is not None and x_right is not None:
                widths.append((y, x_right - x_left))

    if not widths:
        return np.empty((0, 2))
    return np.array(widths)  # shape (N, 2)
