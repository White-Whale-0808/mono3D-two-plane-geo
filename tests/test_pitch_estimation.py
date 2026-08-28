"""Pitch estimation against roads whose grade is known by construction.

These say nothing about accuracy on real imagery — the batch MAE sweep owns
that. What they pin is that the metric stage inverts its own forward model,
and that it declines to answer instead of guessing when the input degenerates.
"""

import numpy as np
import pytest

from libs.inference.lane_fitting import lane_curve
from libs.inference.pitch_estimation import (NearfieldWidthCalibrator,
                                             back_project_widths,
                                             estimate_pitch_from_curves,
                                             estimate_w_real_nearfield,
                                             sample_widths_from_curves)
from tests import synthetic as syn


def _estimate(pitch_deg, method, **kw):
    left, right = syn.road_points(pitch_deg, **kw)
    return estimate_pitch_from_curves(
        lane_curve(left), lane_curve(right),
        syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method=method)


@pytest.mark.parametrize("method", ["windowed", "spline"])
@pytest.mark.parametrize("pitch_deg", [-10.0, -3.0, 0.0, 3.0, 10.0])
def test_recovers_a_constant_grade(method, pitch_deg):
    """A road of constant grade must read back as that grade at every depth."""
    result = _estimate(pitch_deg, method)
    assert result["pitch_at"] is not None
    for z in np.linspace(result["z_visible_min"], result["z_visible_max"], 25):
        assert result["pitch_at"](z) == pytest.approx(pitch_deg, abs=0.1)


def test_reports_the_visible_depth_range():
    result = _estimate(0.0, "windowed", z_near=12.0, z_far=28.0)
    assert result["z_visible_min"] == pytest.approx(12.0, abs=0.1)
    assert result["z_visible_max"] == pytest.approx(28.0, abs=0.1)


def test_back_projection_round_trips():
    """(y, w) -> (z, Y_3d) must invert the forward model exactly."""
    z = np.array([8.0, 15.0, 30.0])
    ys = syn.row_for_depth(z)
    widths = np.column_stack([ys, syn.width_for_depth(z)])
    depths, Y_3d = back_project_widths(widths, syn.F_X, syn.F_Y,
                                       syn.IMG_H, syn.W_REAL)
    assert depths == pytest.approx(z)
    assert Y_3d == pytest.approx(np.full(3, -syn.CAM_H))


def test_sample_density_follows_visible_depth():
    """samples_per_meter is a density: doubling the visible span doubles n."""
    def n_samples(z_near, z_far):
        left, right = syn.road_points(0.0, z_near=z_near, z_far=z_far)
        return len(sample_widths_from_curves(
            lane_curve(left), lane_curve(right), num_samples=80,
            f_x=syn.F_X, w_real=syn.W_REAL, samples_per_meter=6))

    assert n_samples(10.0, 20.0) == pytest.approx(60, abs=1)
    assert n_samples(10.0, 30.0) == pytest.approx(120, abs=1)


@pytest.mark.parametrize("method", ["windowed", "spline"])
def test_declines_when_the_depth_range_is_too_short(method):
    """Below min_valid_range_m there is no slope to measure — say so."""
    result = _estimate(0.0, method, z_near=20.0, z_far=20.2, n=40)
    assert result["pitch_at"] is None
    assert len(result["z_samples"]) == 0


@pytest.mark.parametrize("method", ["windowed", "spline"])
def test_declines_on_a_missing_side(method):
    """One curve absent (junction, abstained side) must not fabricate a pitch."""
    left, _ = syn.road_points(0.0)
    result = estimate_pitch_from_curves(
        lane_curve(left), None, syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method=method)
    assert result["pitch_at"] is None


def _nearfield(theta0_deg=0.0, **kw):
    return estimate_w_real_nearfield(
        syn.nearfield_widths(theta0_deg, **kw),
        syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H)


def test_nearfield_recovers_w_real_on_the_support_plane():
    """Level camera: every row must read back the physical width exactly."""
    result = _nearfield(0.0)
    assert result is not None
    assert result["w_real_med"] == pytest.approx(syn.W_REAL, rel=1e-9)
    assert result["w_real_z0"] == pytest.approx(syn.W_REAL, rel=1e-6)
    assert result["theta0_deg"] == pytest.approx(0.0, abs=1e-6)
    assert result["z_lo"] >= 2.0 and result["z_hi"] <= 5.0


def test_nearfield_separates_mounting_pitch_from_width():
    """A pitched camera trends w(z); intercept and slope split θ0 from w_real."""
    result = _nearfield(1.5)
    assert result["theta0_deg"] == pytest.approx(1.5, abs=0.1)
    assert result["w_real_z0"] == pytest.approx(syn.W_REAL, abs=0.02)
    # the plain median cannot see θ0 and must carry the +w·θ0·z̄/h bias
    assert result["w_real_med"] > syn.W_REAL + 0.2


def test_nearfield_declines_without_enough_near_rows():
    """Curves that stop short of the near window must not fabricate a width."""
    assert _nearfield(0.0, z_near=6.0, z_far=20.0) is None      # all beyond window
    assert _nearfield(0.0, z_near=2.0, z_far=5.0, n=5) is None  # too few rows
    horizon_up = syn.nearfield_widths(0.0)
    horizon_up[:, 0] = syn.IMG_H / 2.0 - 10.0                   # above the horizon
    assert estimate_w_real_nearfield(
        horizon_up, syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H) is None


def _curves(pitch_deg, z_near, z_far):
    left, right = syn.road_points(pitch_deg, z_near=z_near, z_far=z_far)
    return lane_curve(left), lane_curve(right)


def _fed(cal, dist, pitch_deg, z_near=2.0):
    cal.advance_to(dist)
    return cal.update(*_curves(pitch_deg, z_near, 30.0))


def test_calibrator_adopts_holds_and_expires():
    """Full policy over a drive: fallback → run → adopt → hold → expire."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H,
                                   w_real_fallback=3.5)
    # curves that never reach the near window: nothing to measure, fallback
    assert _fed(cal, 0.0, 0.0, z_near=8.0) == pytest.approx(3.5)
    # gate opens, but adoption waits for MIN_RUN metres of open gate
    assert _fed(cal, 0.1, 0.0) == pytest.approx(3.5)
    assert _fed(cal, 0.7, 0.0) == pytest.approx(syn.W_REAL, rel=1e-6)
    # grade knee at the camera (road_points pitches away from the support
    # plane at z=0): a large θ0 trend — reject, hold the adopted value
    held = _fed(cal, 1.0, 8.0)
    assert held == pytest.approx(syn.W_REAL, rel=1e-6)
    assert abs(cal.last_estimate["theta0_deg"]) > cal.theta0_gate_deg
    # still rejected past MAX_HOLD metres: the patch is behind us, fall back
    assert _fed(cal, 0.7 + 5.5, 8.0) == pytest.approx(3.5)


def test_calibrator_ignores_an_isolated_gate_pass():
    """One clean-looking frame between rejects (curvature inflection) must
    not be adopted — the θ0 gate is blind exactly there."""
    cal = NearfieldWidthCalibrator(syn.F_X, syn.F_Y, syn.IMG_H, syn.CAM_H,
                                   w_real_fallback=3.5)
    _fed(cal, 0.0, 8.0)
    _fed(cal, 0.2, 0.0)          # isolated pass: run restarts, span 0 m
    assert _fed(cal, 0.4, 8.0) == pytest.approx(3.5)


def test_declines_on_crossed_curves():
    """Curves that cross give non-positive widths: unphysical, not a pitch."""
    left, right = syn.road_points(0.0)
    result = estimate_pitch_from_curves(
        lane_curve(right), lane_curve(left),   # swapped: right is now inner-left
        syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method="windowed")
    assert result["pitch_at"] is None
