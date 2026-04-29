import cv2
import numpy as np
from libs.inference.lane_fitting import get_x_at_y


def render_piecewise_fits_to_array(resized_image, left_fits, right_fits, widths):
    # 渲染車道擬合線，回傳 BGR ndarray 供 cv2.imshow 使用
    image = np.array(resized_image)
    image_drawn_lane = image.copy()

    for f in left_fits:
        y1, y2 = int(f["y_start"]), int(f["y_end"])
        x1 = int(f["slope"] * y1 + f["intercept"])
        x2 = int(f["slope"] * y2 + f["intercept"])
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

    for f in right_fits:
        y1, y2 = int(f["y_start"]), int(f["y_end"])
        x1 = int(f["slope"] * y1 + f["intercept"])
        x2 = int(f["slope"] * y2 + f["intercept"])
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    for y, w in widths:
        x_l = get_x_at_y(left_fits, y)
        x_r = get_x_at_y(right_fits, y)
        if x_l and x_r:
            cv2.line(image_drawn_lane, (int(x_l), int(y)), (int(x_r), int(y)),
                     (255, 255, 0), 1)

    return cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR)
