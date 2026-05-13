import cv2
import pyelsed
import numpy as np

def detect_lines_with_elsed(masked_road, min_length_near, min_length_far):
    """
    Detect line segments with a y-adaptive length threshold.

    Perspective geometry: a fixed physical lane marking appears shorter the
    further away (smaller y) it is.  Using a single threshold therefore
    over-filters distant / uphill segments while being too lenient near the
    camera.

    The threshold is linearly interpolated between
      - min_length_far  at y = 0   (top of image / vanishing-point region)
      - min_length_near at y = H-1 (bottom of image / near-camera region)
    """
    gray = cv2.cvtColor(masked_road, cv2.COLOR_RGB2GRAY)  # The masked_road image must come from RGB format.
    segments, _ = pyelsed.detect(gray)  # segments is a list of line, size is (N, 4), scores is a list of confidence score for each line, size is (N,)

    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    lengths = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    mid_y = (y1 + y2) / 2

    H = masked_road.shape[0]
    # Linear interpolation: far threshold at top (mid_y=0), near threshold at bottom (mid_y=H)
    adaptive_thr = min_length_far + (min_length_near - min_length_far) * (mid_y / H)
    segments = segments[lengths >= adaptive_thr]  # Discard short segments that typically come from mask boundary noise

    return segments
