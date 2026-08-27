"""Depth-continuity guard (WWH-15).

Guards the one case the photometric check cannot: real paint beyond a crest,
which is a DISCONNECTED road section and must never be bridged into the near
chain. The hard part is not detecting the jump — it is not firing on a large
row gap in continuous road, where the depth also advances a long way.
"""

import numpy as np
import pytest

from libs.inference.lane_fitting import _ZJUMP_ABS_M, truncate_at_depth_jump
from tests import synthetic as syn


def _truncate(rows_and_depths):
    left, right = syn.paired_rows(rows_and_depths)
    return truncate_at_depth_jump(left, right, syn.F_X, syn.W_REAL, syn.IMG_H)


def _depths(left, right):
    """Recover per-row depth from a pair of chains, near first."""
    xl = {int(round(y)): x for x, y in left}
    xr = {int(round(y)): x for x, y in right}
    return [syn.F_X * syn.W_REAL / (xr[y] - xl[y])
            for y in sorted(set(xl) & set(xr), reverse=True)]


def test_a_smooth_profile_is_left_alone():
    profile = [(syn.row_for_depth(z), z) for z in np.arange(5.0, 20.0, 0.5)]
    left, right = _truncate(profile)
    assert len(left) == len(profile)
    assert len(right) == len(profile)


def test_a_large_row_gap_on_continuous_road_is_not_a_jump():
    """Regression: down_hile 209.

    Rows 400 and 316 with depths 2.5 and 6.0 m. The 3.5 m step clears the
    absolute gate (_ZJUMP_ABS_M = 3.0), and the first version truncated here
    and lost the frame. But 84 rows of continuous road genuinely advance that
    far: the local-plane extrapolation z1·(y1-cy)/(y2-cy) is exactly 6.0, so
    a continuous surface reaches it and there is nothing hidden.
    """
    y1, z1 = 400.0, 2.5
    y2 = syn.CY + (y1 - syn.CY) * z1 / 6.0        # z_exp == 6.0 by construction
    z2 = 6.0
    assert z2 - z1 > _ZJUMP_ABS_M                 # the absolute gate IS tripped
    left, right = _truncate([(y1, z1), (y2, z2)])
    assert len(left) == 2 and len(right) == 2


def test_an_occlusion_jump_truncates_both_chains():
    """Same rows, but a depth no continuous surface could reach."""
    y1, z1 = 400.0, 2.5
    y2 = syn.CY + (y1 - syn.CY) * z1 / 6.0        # extrapolation still says 6.0
    left, right = _truncate([(y1, z1), (y2, 25.0)])
    assert len(left) == 1 and len(right) == 1
    assert left[0][1] == pytest.approx(y1)
    assert right[0][1] == pytest.approx(y1)


def test_unpaired_points_inside_the_gap_are_cut_too():
    """Rows above the jump are unverified even where only one side has a point."""
    y1, z1 = 400.0, 2.5
    y2 = syn.CY + (y1 - syn.CY) * z1 / 6.0
    left, right = syn.paired_rows([(y1, z1), (y2, 25.0)])
    dangling = np.array([[500.0, (y1 + y2) / 2.0]])     # left-only, in the gap
    left = np.vstack([left, dangling])

    left_out, right_out = truncate_at_depth_jump(
        left, right, syn.F_X, syn.W_REAL, syn.IMG_H)
    assert len(left_out) == 1 and len(right_out) == 1
    assert left_out[0][1] == pytest.approx(y1)


def test_the_near_side_of_the_jump_is_kept_not_the_far_side():
    """The near chain is the verified one; the far section is what is dropped."""
    profile = [(syn.row_for_depth(z), z) for z in (5.0, 6.0, 7.0, 8.0)]
    y_far = syn.row_for_depth(9.0)
    left, right = _truncate(profile + [(y_far, 40.0)])
    assert _depths(left, right) == pytest.approx([5.0, 6.0, 7.0, 8.0])


def test_an_empty_side_is_returned_unchanged():
    left, right = syn.paired_rows([(400.0, 5.0)])
    out_l, out_r = truncate_at_depth_jump(
        left, np.empty((0, 2)), syn.F_X, syn.W_REAL, syn.IMG_H)
    assert len(out_l) == 1 and len(out_r) == 0
