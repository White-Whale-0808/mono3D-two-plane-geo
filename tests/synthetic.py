"""Synthetic pinhole scenes.

Everything here is built from the forward model the pipeline inverts, so the
expected answers are known analytically rather than recorded from a run:

    z(y)     = f_y·h / (y - cy)          flat road at camera height h
    Y_3d     = -z·(y - cy) / f_y         camera-frame height of the road point
    w_px(z)  = f_x·w_real / z            inner-edge-to-inner-edge pixel width

The calibration below is this project's dataset (see the config comments):
1024x512 after resize, f_x/f_y = 512/455, camera 1.08 m up, w_real 3.25 m.
"""

import numpy as np

F_X = 512.0
F_Y = 455.0
CAM_H = 1.08
W_REAL = 3.25
IMG_W = 1024
IMG_H = 512
CX = IMG_W / 2.0
CY = IMG_H / 2.0

# Road grey level and painted-marking grey level for the photometric scenes.
ROAD_GRAY = 80
PAINT_GRAY = 200


def road_points(pitch_deg=0.0, z_near=10.0, z_far=30.0, n=200,
                f_x=F_X, f_y=F_Y, camera_height=CAM_H, w_real=W_REAL,
                image_height=IMG_H, image_width=IMG_W):
    """Inner-edge points of both lane lines for a straight road of constant grade.

    A road of grade θ has Y_3d(z) = -h + z·tanθ, so `estimate_pitch_*` must
    report exactly θ — arctan of the slope of Y_3d over z.

    The depth range defaults to 10-30 m deliberately: it keeps every sample
    inside the estimator's IQR width filter, so a test that fails is telling
    you about the estimator and not about sample rejection.

    Returns (left_points, right_points), each an (n, 2) array of (x, y).
    """
    cx, cy = image_width / 2.0, image_height / 2.0
    z = np.linspace(z_near, z_far, n)
    Y = -camera_height + z * np.tan(np.radians(pitch_deg))
    y = cy - f_y * Y / z
    w_px = f_x * w_real / z
    left = np.column_stack([cx - w_px / 2.0, y])
    right = np.column_stack([cx + w_px / 2.0, y])
    return left, right


def row_for_depth(z, f_y=F_Y, camera_height=CAM_H, image_height=IMG_H):
    """Image row a flat road at depth z projects to."""
    return image_height / 2.0 + f_y * camera_height / z


def width_for_depth(z, f_x=F_X, w_real=W_REAL):
    """Inner-edge-to-inner-edge pixel width at depth z."""
    return f_x * w_real / z


def paired_rows(rows_and_depths, cx=CX, f_x=F_X, w_real=W_REAL):
    """Left/right chains that realise a given (row, depth) sequence exactly.

    `rows_and_depths` is an iterable of (y, z). Used to hand-build the depth
    profiles that the depth-continuity guard has to classify.
    """
    left, right = [], []
    for y, z in rows_and_depths:
        half = width_for_depth(z, f_x, w_real) / 2.0
        left.append((cx - half, float(y)))
        right.append((cx + half, float(y)))
    return np.array(left, dtype=float), np.array(right, dtype=float)


def road_image(image_width=IMG_W, image_height=IMG_H):
    """Uniform road with no markings at all."""
    return np.full((image_height, image_width, 3), ROAD_GRAY, dtype=np.uint8)


def paint_stripe(image, x_inner, rows, is_left, stripe_px=6,
                 gray=PAINT_GRAY):
    """Paint a bounded bright stripe lying OUTWARD of the inner edge x_inner.

    This is what the photometric check is looking for: intensity rises at the
    edge, holds across `stripe_px`, then falls back to road level.
    """
    out = -1 if is_left else 1
    for y in rows:
        yi = int(round(y))
        for j in range(1, stripe_px + 1):
            image[yi, int(round(x_inner)) + out * j] = gray
    return image


def paint_plateau(image, x_inner, rows, is_left, gray=PAINT_GRAY):
    """Bright region that never falls back to road level going outward.

    A far-field bright surface or the lit side of a shadow boundary — bright
    at the edge like paint, but unbounded, so the far-drop probe must reject
    it. This is the case that broke when the bright-peak window was derived
    from the projected stripe width instead of being fixed (down_hile 157).
    """
    out = -1 if is_left else 1
    w = image.shape[1]
    for y in rows:
        yi = int(round(y))
        xi = int(round(x_inner))
        xs = range(xi - 1, -1, -1) if out < 0 else range(xi + 1, w)
        for x in xs:
            image[yi, x] = gray
    return image
