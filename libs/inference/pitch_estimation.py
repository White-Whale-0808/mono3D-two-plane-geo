import numpy as np
from scipy.stats import theilslopes

def estimate_pitch_from_widths(widths, f_x, f_y, image_height, w_real):
    # Remove width outliers via IQR before computing depth,
    # so that bad lane-fitting samples in some bands don't skew the pitch estimate.
    w = widths[:, 1]
    q1, q3 = np.percentile(w, [25, 75])
    iqr = q3 - q1
    valid = (w >= q1 - 1.5 * iqr) & (w <= q3 + 1.5 * iqr)
    widths = widths[valid]

    depths = f_x * w_real / widths[:, 1]

    center_y = image_height / 2
    Y_3d = -depths * (widths[:, 0] - center_y) / f_y

    # coeffs = np.polyfit(depths, Y_3d, deg=1)
    # slope = coeffs[0]
    result = theilslopes(Y_3d, depths)  # Use Theil-Sen estimator to be more robust to outliers
    pitch_rad = np.arctan(result.slope)
    pitch_deg = np.degrees(pitch_rad)
    return pitch_deg