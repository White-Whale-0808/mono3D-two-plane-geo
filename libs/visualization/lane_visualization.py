import cv2
import numpy as np
from libs.inference.lane_fitting import get_x_at_y

def draw_lane_lines(resized_image, left_lines, right_lines, save_path):
    image = np.array(resized_image)
    image_drawn_lane = image.copy()  # Python is call-by-reference. 

    for line in left_lines:
        x1, y1, x2, y2 = line
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)  # cv2.LINE_AA for anti-aliased lines

    for line in right_lines:
        x1, y1, x2, y2 = line
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

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

# """
# Used to visualize the lane seed, which is the lane segmentation method 1
# """
# def draw_lane_seed(resized_image, left_lines, right_lines, save_path):
#     image = np.array(resized_image)
#     image_drawn_lane = image.copy() 

#     x1, y1, x2, y2 = left_lines["seg"]
#     cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)  # cv2.LINE_AA for anti-aliased lines

#     x1, y1, x2, y2 = right_lines["seg"]
#     cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

#     cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))

# """
# Used to visualize the k-means clustering result, which is the lane segmentation method 2
# """
# def draw_kmeans_clusters(resized_image, left_clusters, right_clusters, save_path):
#     image = np.array(resized_image)
#     image_drawn_lane = image.copy()

#     left_colors  = [(255, 0, 0), (255, 255, 0)]  # Red and Yellow for left lane clusters
#     right_colors = [(0, 0, 255), (0, 255, 0)]  # Blue and Green for right lane clusters    

#     for i, cluster in enumerate(left_clusters):
#         for line in cluster:
#             x1, y1, x2, y2 = line["seg"]
#             cv2.line(image_drawn_lane, (x1, y1), (x2, y2), left_colors[i], 2, cv2.LINE_AA)

#     for i, cluster in enumerate(right_clusters):
#         for line in cluster:
#             x1, y1, x2, y2 = line["seg"]
#             cv2.line(image_drawn_lane, (x1, y1), (x2, y2), right_colors[i], 2, cv2.LINE_AA)

#     cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))

def draw_piecewise_fits(resized_image, left_fits, right_fits, widths, save_path):
    image = np.array(resized_image)
    image_drawn_lane = image.copy()  # Python is call-by-reference. 

    for f in left_fits:
        y1 = int(f["y_start"])
        y2 = int(f["y_end"])
        x1 = int(f["slope"] * y1 + f["intercept"])
        x2 = int(f["slope"] * y2 + f["intercept"])
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
    
    for f in right_fits:
        y1 = int(f["y_start"])
        y2 = int(f["y_end"])
        x1 = int(f["slope"] * y1 + f["intercept"])
        x2 = int(f["slope"] * y2 + f["intercept"])
        cv2.line(image_drawn_lane, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    for y, w in widths:
            x_l = get_x_at_y(left_fits, y)
            x_r = get_x_at_y(right_fits, y)
            if x_l and x_r:
                cv2.line(image_drawn_lane, (int(x_l), int(y)), (int(x_r), int(y)),
                         (255, 255, 0), 1)
    cv2.imwrite(save_path, cv2.cvtColor(image_drawn_lane, cv2.COLOR_RGB2BGR))
