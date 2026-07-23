import cv2
import numpy as np
from pathlib import Path

# per-fragment display palette for the step visualization (cycled)
_FRAG_PALETTE = [(255, 90, 90), (90, 200, 255), (255, 200, 0), (180, 120, 255),
                 (0, 220, 140), (255, 140, 60), (120, 160, 255), (240, 240, 90)]


def _dim(image_rgb, factor=0.45):
    return (np.asarray(image_rgb).astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def _legend(img, entries):
    x, y = 8, 20
    for text, color in entries:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += 18


def _save_rgb(img, path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _dashed(img, p0, p1, color, thickness=2, dash=6):
    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = max(int(length / dash), 1)
    for i in range(0, n, 2):
        a = (int(round(x0 + (x1 - x0) * i / n)), int(round(y0 + (y1 - y0) * i / n)))
        b = (int(round(x0 + (x1 - x0) * min(i + 1, n) / n)),
             int(round(y0 + (y1 - y0) * min(i + 1, n) / n)))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_line_segments(resized_image, segments, save_path):
    image = np.array(resized_image)
    image_drawn_lane = image.copy()  # Python is call-by-reference.

    for line in segments:
        x1, y1, x2, y2 = line
        cv2.line(image_drawn_lane, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2, cv2.LINE_AA)  # cv2.LINE_AA for anti-aliased lines
    cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))

def draw_lane_lines(resized_image, left_lines, right_lines, save_path):
    """Tracked lane-line groups from lane_segmentation: every kept segment,
    left red / right green (whole marking group — the lane_fitting INPUT)."""
    image = np.array(resized_image)
    image_drawn_lane = image.copy()  # Python is call-by-reference.

    for lines, color in ((left_lines, (255, 0, 0)), (right_lines, (0, 255, 0))):
        for x1, y1, x2, y2 in lines:
            cv2.line(image_drawn_lane, (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))

def create_overlay(resized_image, pred_mask, alpha, save_path):
    image = np.array(resized_image)
    overlay = image.astype(np.float32).copy()

    # Alpha blending. We only apply the red color to the road area, and keep the non-road area unchanged.
    overlay[pred_mask == 1] = (
        alpha * np.array([255, 0, 0]) +
        (1 - alpha) * overlay[pred_mask == 1]
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)  # Ensure pixel values are valid after blending

    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_lane_fitting_steps(resized_image, left_lines, right_lines, out_dir):
    """Step-by-step lane_fitting visualization into out_dir (one PNG per
    stage): 00 input segment groups, 01 shadowing, 02 dense inner envelope,
    03 fragments + junction groups, 04 sub-pixel refinement, 05 final curves
    with bridged gaps. Uses inner_chain_points(return_debug=True) — no
    mirrored logic. Returns out_dir."""
    from libs.inference.lane_fitting import (inner_chain_points,
                                             refine_inner_points, lane_curve)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    side_color = {"L": (255, 0, 0), "R": (0, 255, 0)}
    sides = {}
    for name, lines, is_left in (("L", left_lines, True), ("R", right_lines, False)):
        pts, dbg = inner_chain_points(lines, is_left, return_debug=True)
        ref = refine_inner_points(resized_image, pts, is_left)
        sides[name] = dict(lines=lines, pts=pts, dbg=dbg, ref=ref,
                           curve=lane_curve(ref))

    def _seg(img, seg, color, thickness=2):
        x1, y1, x2, y2 = seg
        cv2.line(img, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color, thickness, cv2.LINE_AA)

    # -- 00: input tracked groups --------------------------------------------
    img = np.asarray(resized_image).copy()
    for name, s in sides.items():
        for seg in s["lines"]:
            _seg(img, seg, side_color[name])
    _legend(img, [(f"input tracked segments  L={len(sides['L']['lines'])} "
                   f"R={len(sides['R']['lines'])} (whole marking groups)",
                   (255, 255, 255))])
    _save_rgb(img, out / "00_input_segments.png")

    # -- 01: shadowing --------------------------------------------------------
    img = _dim(resized_image)
    n_shadow = 0
    for name, s in sides.items():
        for seg in s["dbg"]["shadowed"]:
            _seg(img, seg, (255, 160, 40), 2)
            n_shadow += 1
        for seg in s["dbg"]["kept_segments"]:
            _seg(img, seg, side_color[name], 2)
    _legend(img, [
        ("kept segments (L)", side_color["L"]),
        ("kept segments (R)", side_color["R"]),
        (f"shadowed = outer paint edge, excluded ({n_shadow})", (255, 160, 40)),
        ("outer-only rows become GAPS, never fallback", (255, 255, 255)),
    ])
    _save_rgb(img, out / "01_shadowing.png")

    # -- 02: dense per-row inner envelope -------------------------------------
    img = _dim(resized_image)
    for name, s in sides.items():
        rows = s["dbg"]["rows"]
        for (ya, xa), (yb, xb) in zip(rows, rows[1:]):
            if ya - yb == 1:  # rows are y-descending
                cv2.line(img, (int(round(xa)), int(round(ya))),
                         (int(round(xb)), int(round(yb))),
                         side_color[name], 1, cv2.LINE_AA)
    _legend(img, [
        (f"per-row inner envelope: L={len(sides['L']['dbg']['rows'])} "
         f"R={len(sides['R']['dbg']['rows'])} rows (1 px each)", (255, 255, 255)),
        ("uncovered rows stay gaps", (255, 255, 255)),
    ])
    _save_rgb(img, out / "02_envelope.png")

    # -- 03: fragments + junction groups --------------------------------------
    img = _dim(resized_image)
    n_frag = n_dropped = 0
    for s in sides.values():
        groups, best = s["dbg"]["groups"], s["dbg"]["best_group"]
        k = 0
        for gi, group in enumerate(groups):
            for frag in group:
                kept = gi == best
                color = _FRAG_PALETTE[k % len(_FRAG_PALETTE)] if kept else (255, 60, 60)
                k += 1
                n_frag += 1
                n_dropped += 0 if kept else 1
                pts = np.array([(x, y) for y, x in frag])
                if len(pts) == 1:
                    cv2.circle(img, (int(round(pts[0, 0])), int(round(pts[0, 1]))),
                               2, color, -1)
                else:
                    poly = pts.round().astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(img, [poly], False, color, 2 if kept else 1,
                                  cv2.LINE_AA)
    _legend(img, [
        (f"fragments: {n_frag} total (colors cycle within kept group)", (255, 255, 255)),
        (f"red thin = dropped by junction consistency ({n_dropped})", (255, 60, 60)),
        ("split at coverage gaps / 1-row x-jump > 4px", (255, 255, 255)),
    ])
    _save_rgb(img, out / "03_fragments.png")

    # -- 04: sub-pixel refinement ---------------------------------------------
    img = _dim(resized_image)
    stats = []
    for name, s in sides.items():
        for x, y in s["pts"].reshape(-1, 2):
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (160, 160, 160), 1)
        for x, y in s["ref"].reshape(-1, 2):
            cv2.circle(img, (int(round(x)), int(round(y))), 1, side_color[name], -1)
        if len(s["pts"]):
            dx = s["ref"][:, 0] - s["pts"][:, 0]
            stats.append(f"{name}: |dx| mean {np.abs(dx).mean():.2f}px "
                         f"max {np.abs(dx).max():.2f}px")
    _legend(img, [
        ("hollow gray = ELSED prior, filled = refined edge", (255, 255, 255)),
        ("gradient peak nearest to prior, +-3px, parabola sub-pixel",
         (255, 255, 255)),
    ] + [(t, (255, 255, 255)) for t in stats])
    _save_rgb(img, out / "04_refine.png")

    # -- 05: final curves, bridged gaps dashed --------------------------------
    img = _dim(resized_image)
    for name, s in sides.items():
        curve = s["curve"]
        if curve is None:
            continue
        ys, xs = curve["y"], curve["x"]
        for i in range(len(ys) - 1):
            p0 = (xs[i], ys[i])
            p1 = (xs[i + 1], ys[i + 1])
            if ys[i + 1] - ys[i] > 1.5:  # no chain point in between -> bridge
                _dashed(img, p0, p1, (255, 255, 0), 2)
            else:
                cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                         (int(round(p1[0])), int(round(p1[1]))),
                         side_color[name], 2, cv2.LINE_AA)
    _legend(img, [
        ("lane_curve: polyline through refined points", (255, 255, 255)),
        ("solid = on chain points (L/R)", (255, 255, 255)),
        ("yellow dashed = gap bridged linearly (model, not data)", (255, 255, 0)),
    ])
    _save_rgb(img, out / "05_curve.png")
    return str(out)


def draw_lane_curves(resized_image, left_curve, right_curve, save_path):
    """Draw the two continuous lane curves (lane_fitting.lane_curve polylines).

    Width sampling lives in pitch_estimation, so this stage's visualization
    shows only the lane_fitting output: one gap-bridged curve per side."""
    image = np.array(resized_image)
    image_drawn_lane = image.copy()  # Python is call-by-reference.

    for curve, color in ((left_curve, (255, 0, 0)), (right_curve, (0, 255, 0))):
        if curve is None:
            continue
        pts = np.column_stack([curve["x"], curve["y"]])
        pts = pts.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_drawn_lane, [pts], False, color, 2, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))
