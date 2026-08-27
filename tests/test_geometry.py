"""CameraGeometry: the projection model every stage derives its thresholds from.

The expected values come from the pinhole equations, not from a recorded run.
"""

import numpy as np
import pytest

from libs.inference.geometry import (CameraGeometry, _GRADE_RAMP_SPAN,
                                     _GRADE_RAMP_Z0, _MAX_GRADE_DEG)
from tests import synthetic as syn


@pytest.fixture
def geom():
    return CameraGeometry(syn.F_X, syn.F_Y, syn.CAM_H, syn.W_REAL,
                          syn.IMG_W, syn.IMG_H)


@pytest.mark.parametrize("z", [3.0, 6.0, 12.0, 18.0])
def test_z_at_inverts_the_projection(geom, z):
    """z_at(row a flat road at z projects to) == z, below the clamp."""
    assert geom.z_at(syn.row_for_depth(z)) == pytest.approx(z, rel=1e-9)


def test_z_at_saturates_beyond_the_clamp_depth(geom):
    """The clamp bites at ~19 m, well inside the working range.

    min_y_margin caps 1/(y-cy) at 1/(0.05·image_height) = 1/25.6 px, so
    z_at tops out at f_y·h/25.6 ≈ 19.2 m — anything further reads back as
    that same value. This is deliberate (z_at is a FLAT-ground approximation
    and drifts on the second plane), and it is why thresholds derived from
    it stay conservative at range rather than growing without bound.
    """
    z_cap = syn.F_Y * syn.CAM_H / geom.min_dy
    assert z_cap == pytest.approx(19.2, abs=0.1)
    assert geom.z_at(syn.row_for_depth(30.0)) == pytest.approx(z_cap)
    assert geom.z_at(syn.row_for_depth(45.0)) == pytest.approx(z_cap)


def test_z_at_is_clamped_at_the_horizon(geom):
    """Rows at or above the flat horizon saturate instead of blowing up."""
    horizon_z = syn.F_Y * syn.CAM_H / geom.min_dy
    assert geom.z_at(geom.cy) == pytest.approx(horizon_z)
    assert geom.z_at(geom.cy - 50) == pytest.approx(horizon_z)
    assert np.isfinite(geom.z_at(geom.cy - 50))


def test_z_valid_marks_the_clamped_rows(geom):
    assert geom.z_valid(geom.cy + geom.min_dy + 1.0)
    assert not geom.z_valid(geom.cy + geom.min_dy - 1.0)
    assert not geom.z_valid(geom.cy - 10)


@pytest.mark.parametrize("z", [3.0, 8.0, 15.0])
def test_lane_px_is_the_projected_lane_width(geom, z):
    y = syn.row_for_depth(z)
    assert geom.lane_px(y) == pytest.approx(syn.F_X * syn.W_REAL / z)


def test_no_grade_slack_within_the_ego_plane(geom):
    """Inside _GRADE_RAMP_Z0 the ego stands on the road: z_min == z_at."""
    y = syn.row_for_depth(_GRADE_RAMP_Z0 - 1.0)
    assert geom.z_min(y) == pytest.approx(geom.z_at(y))


def test_grade_slack_is_fully_active_beyond_the_ramp(geom):
    """Past the ramp, z_min is the +_MAX_GRADE_DEG plane's depth."""
    z_flat = _GRADE_RAMP_Z0 + _GRADE_RAMP_SPAN + 10.0
    y = syn.row_for_depth(z_flat)
    expected = syn.F_Y * syn.CAM_H / (
        (y - geom.cy) + syn.F_Y * np.tan(np.radians(_MAX_GRADE_DEG)))
    assert geom.z_min(y) == pytest.approx(expected)


def test_z_min_never_exceeds_z_at(geom):
    """z_min is a LOWER bound on the true depth at every row."""
    for y in np.linspace(geom.cy + geom.min_dy, syn.IMG_H - 1, 60):
        assert geom.z_min(y) <= geom.z_at(y) + 1e-9


def test_lane_px_max_bounds_lane_px(geom):
    """The worst-case (uphill) lane width is never narrower than nominal."""
    for y in np.linspace(geom.cy + geom.min_dy, syn.IMG_H - 1, 60):
        assert geom.lane_px_max(y) >= geom.lane_px(y) - 1e-9
