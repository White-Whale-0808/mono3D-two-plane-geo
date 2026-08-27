import numpy as np

"""
Shared flat-ground pinhole projection model.

Extracted from lane_segmentation (which derives its tracking thresholds from
it) once paint_evidence needed the same z_min-bounded probe windows — the
camera model is stage-neutral, so it lives here instead of one stage
importing another's internals.
"""

# Grade-uncertainty constants (used by z_min / lane_px_max bounds)
_MAX_GRADE_DEG   = 15.0   # worst-case road grade the thresholds must absorb
_GRADE_RAMP_Z0   = 6.0    # within 6 m the road is the ego plane: no grade slack
_GRADE_RAMP_SPAN = 6.0    # grade slack ramps to full between 6 m and 12 m


class CameraGeometry:
    """Projection helpers.

    z_at(y) assumes a flat road. The road may however be tilted by up to
    ±_MAX_GRADE_DEG: a plane of grade θ passing under the camera satisfies
        y - cy = f_y*h/z - f_y*tanθ   =>   z = f_y*h / ((y-cy) + f_y*tanθ)
    so z_min(y) (taken at +θ_max, uphill) lower-bounds the true depth, and is
    also finite for rows ABOVE the flat horizon — exactly where uphill road
    appears. Pixel windows derived with z_min are valid for any grade within
    the bound; that is the principled replacement for hand-tuned margins.
    """

    def __init__(self, f_x, f_y, camera_height, w_real,
                 image_width, img_height, min_y_margin=0.05):
        self.f_x = float(f_x)
        self.f_y = float(f_y)
        self.h = float(camera_height)
        self.w = float(w_real)
        self.cx = image_width / 2.0
        self.cy = img_height / 2.0
        self.min_dy = min_y_margin * img_height  # clamp near/above horizon
        self.grade_dy = self.f_y * np.tan(np.radians(_MAX_GRADE_DEG))

    def z_at(self, y):
        """Flat-ground (nominal) depth of image row y, clamped at the horizon."""
        return self.f_y * self.h / max(y - self.cy, self.min_dy)

    def z_min(self, y):
        """Lower depth bound at row y over grades within ±_MAX_GRADE_DEG.

        The ego vehicle stands on the near plane, so rows closer than
        _GRADE_RAMP_Z0 carry no grade uncertainty; the slack ramps in
        linearly and is fully active beyond _GRADE_RAMP_Z0 + _GRADE_RAMP_SPAN
        (where a second, tilted plane may appear).
        """
        z_flat = self.z_at(y)
        w = min(max((z_flat - _GRADE_RAMP_Z0) / _GRADE_RAMP_SPAN, 0.0), 1.0)
        return self.f_y * self.h / max(y - self.cy + w * self.grade_dy, self.min_dy)

    def lane_px(self, y):
        """Nominal (flat-ground) pixel width of one real lane at row y."""
        return self.f_x * self.w / self.z_at(y)

    def lane_px_max(self, y):
        """Upper bound of the lane pixel width at row y (worst-case uphill)."""
        return self.f_x * self.w / self.z_min(y)

    def z_valid(self, y):
        """True when flat-ground z(y) is not saturated by the horizon clamp."""
        return (y - self.cy) > self.min_dy
