"""Pitch estimation against roads whose grade is known by construction.

These say nothing about accuracy on real imagery — the batch MAE sweep owns
that. What they pin is that the metric stage inverts its own forward model,
and that it declines to answer instead of guessing when the input degenerates.
"""

import numpy as np
import pytest

from libs.inference.lane_fitting import lane_curve
from libs.inference.pitch_estimation import (back_project_widths,
                                             estimate_pitch_from_curves,
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


def test_declines_on_crossed_curves():
    """Curves that cross give non-positive widths: unphysical, not a pitch."""
    left, right = syn.road_points(0.0)
    result = estimate_pitch_from_curves(
        lane_curve(right), lane_curve(left),   # swapped: right is now inner-left
        syn.F_X, syn.F_Y, syn.IMG_H, syn.W_REAL,
        num_samples=80, samples_per_meter=6, method="windowed")
    assert result["pitch_at"] is None
