"""Paint-evidence guards (WWH-15).

Each test here corresponds to a case that cost a full three-dataset sweep to
find. The run-length rule, the leading-drop exception and the fixed bright-peak
window are all load-bearing, and none of them is obvious from the code alone.
"""

import numpy as np
import pytest

from libs.inference.paint_evidence import (_FAR_CAP_PX, _PEAK_PX,
                                           _TRUNC_MIN_RUN, _drop_window,
                                           filter_paint_segments,
                                           truncate_at_evidence_break)
from libs.inference.geometry import CameraGeometry
from tests import synthetic as syn

CAM = (syn.F_X, syn.F_Y, syn.CAM_H, syn.W_REAL)
# Rows well inside the image, ordered near (large y) to far (small y).
ROWS = np.arange(460, 300, -4, dtype=float)
X_INNER = 300.0


def _chain(rows, x=X_INNER):
    return np.column_stack([np.full(len(rows), x), rows])


def _truncate(image, rows, is_left=True):
    return truncate_at_evidence_break(image, _chain(rows), is_left, *CAM)


def test_a_fully_painted_chain_survives_intact():
    image = syn.paint_stripe(syn.road_image(), X_INNER, ROWS, is_left=True)
    kept = _truncate(image, ROWS)
    assert len(kept) == len(ROWS)


def test_an_unpainted_chain_is_rejected_entirely():
    """No paint anywhere: every point is a leading failure and gets dropped."""
    kept = _truncate(syn.road_image(), ROWS)
    assert len(kept) == 0


def test_a_sustained_break_cuts_the_far_tail():
    """The crest-occlusion signature: paint, then a long run of non-paint."""
    n_paint = 12
    image = syn.paint_stripe(syn.road_image(), X_INNER, ROWS[:n_paint],
                             is_left=True)
    kept = _truncate(image, ROWS)
    assert len(kept) == n_paint
    assert kept[:, 1].min() == ROWS[n_paint - 1]


def test_an_isolated_failure_does_not_cut():
    """Normal frames show single dropouts; only a RUN means occlusion."""
    painted = [r for i, r in enumerate(ROWS) if i != 8]
    image = syn.paint_stripe(syn.road_image(), X_INNER, painted, is_left=True)
    kept = _truncate(image, ROWS)
    assert len(kept) == len(ROWS)


def test_a_run_just_short_of_the_threshold_does_not_cut():
    gap = set(range(8, 8 + _TRUNC_MIN_RUN - 1))
    painted = [r for i, r in enumerate(ROWS) if i not in gap]
    image = syn.paint_stripe(syn.road_image(), X_INNER, painted, is_left=True)
    kept = _truncate(image, ROWS)
    assert len(kept) == len(ROWS)


def test_leading_failures_are_dropped_not_treated_as_a_break():
    """Regression: uphile 387/396, down_hile 145/146.

    A dark strip at the image bottom made the chain's NEAR end fail, and the
    first version cut there — which is cut=0, i.e. the whole side thrown away.
    An occlusion boundary lies BEYOND verified road, so a failing run before
    any pass is a local artifact: drop those points, keep the rest.
    """
    n_lead = _TRUNC_MIN_RUN + 3          # long enough to trip the run rule
    image = syn.paint_stripe(syn.road_image(), X_INNER, ROWS[n_lead:],
                             is_left=True)
    kept = _truncate(image, ROWS)
    assert len(kept) == len(ROWS) - n_lead
    assert kept[:, 1].max() == ROWS[n_lead]


def test_a_bright_plateau_is_not_paint():
    """Regression: down_hile 157.

    Paint is a BOUNDED ridge. A bright surface that never falls back to road
    level going outward (far field beyond a crest, the lit side of a shadow
    boundary) is bright at the edge but must still fail. It does so only
    because the bright-peak window is fixed at 1.._PEAK_PX — deriving it from
    the projected stripe width inflates it to ~23 px near the horizon and the
    window swallows the plateau.
    """
    image = syn.paint_plateau(syn.road_image(), X_INNER, ROWS, is_left=True)
    assert len(_truncate(image, ROWS)) == 0


def test_only_the_far_drop_probe_scales_with_geometry():
    """The two windows have deliberately different natures.

    The far-drop probe must clear the whole stripe, so it IS derived from the
    projected stripe width: wide near the camera, narrow at range. The
    bright-peak window is a fixed pixel count, because the probe sits ON the
    edge and the bright side starts immediately whatever the stripe width.
    Deriving the peak window the same way is what broke down_hile 157.
    """
    assert isinstance(_PEAK_PX, int)

    geom = CameraGeometry(*CAM, syn.IMG_W, syn.IMG_H)
    near_lo, near_hi = _drop_window(geom, 480.0)     # z ≈ 2 m
    far_lo, far_hi = _drop_window(geom, 270.0)       # beyond the clamp depth

    assert near_lo > far_lo                          # stripe shrinks with depth
    assert near_hi <= _FAR_CAP_PX and far_hi <= _FAR_CAP_PX
    # The far window is bounded away from zero by the grade slack in z_min:
    # a +15° plane puts the stripe closer than the flat model says.
    assert far_lo >= 2 + 2


def test_the_segment_gate_keeps_paint_and_drops_a_shadow_boundary():
    """Layer 1: this is what lets the tracker re-seed on the true line."""
    rows = ROWS[:40]
    image = syn.road_image()
    syn.paint_stripe(image, X_INNER, rows, is_left=True)
    syn.paint_plateau(image, 600.0, rows, is_left=False)

    paint_seg = [X_INNER, rows[0], X_INNER, rows[-1]]
    shadow_seg = [600.0, rows[0], 600.0, rows[-1]]
    kept = filter_paint_segments(image, np.array([paint_seg, shadow_seg]), *CAM)

    assert len(kept) == 1
    assert kept[0][0] == pytest.approx(X_INNER)


def test_an_empty_input_is_returned_unchanged():
    empty = np.empty((0, 4))
    assert len(filter_paint_segments(syn.road_image(), empty, *CAM)) == 0
    assert len(truncate_at_evidence_break(
        syn.road_image(), np.empty((0, 2)), True, *CAM)) == 0
