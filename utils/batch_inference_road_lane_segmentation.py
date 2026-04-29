import yaml
from pathlib import Path
import pandas as pd
from libs.inference.road_segmentation import load_pidnet
from libs.inference.pipeline import infer_one
import traceback

with open("config/inference_road_lane_segmentation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device                    = config["model"]["device"]
model_name                = config["model"]["model_name"]
weight_path               = config["model"]["weight_path"]
image_batch_path          = config["input"]["image_batch_path"]
image_batch_size          = config["input"]["image_batch_size"]
resize_size               = tuple(config["input"]["resize_size"])
min_slope                 = config["lane_segmentation"]["min_slope"]
min_segment_length        = config["lane_segmentation"]["min_segment_length"]
lane_band_tolerance       = config["lane_segmentation"]["lane_band_tolerance"]
num_bands                 = config["lane_fitting"]["num_bands"]
num_samples               = config["lane_fitting"]["num_samples"]
extra_points_per_segment  = config["lane_fitting"]["extra_points_per_segment"]
f_x                       = config["pitch_estimation"]["f_x"]
f_y                       = config["pitch_estimation"]["f_y"]
w_real                    = config["pitch_estimation"]["w_real"]
input_csv                 = config["csv_io"]["input_dir"]
output_csv                = config["csv_io"]["output_dir"]

IMG_FMT  = "{:06d}.png"   

def main():
    df = pd.read_csv(input_csv)
    model = load_pidnet(model_name, weight_path, device)
    
    n_total          = len(df)
    pred_deg         = [pd.NA] * n_total
    n_skip_pipeline  = 0
    
    image_batch_dir = Path(image_batch_path)
    for i, row in enumerate(df.itertuples(index=False)):
        frame_id = int(row.frame_id)
        image_path = image_batch_dir / IMG_FMT.format(frame_id)

        try:
            pitch = infer_one(
                model, str(image_path),
                device=device, resize_size=resize_size,
                min_slope=min_slope, min_segment_length=min_segment_length,
                lane_band_tolerance=lane_band_tolerance,
                extra_points_per_segment=extra_points_per_segment,
                num_bands=num_bands, num_samples=num_samples,
                f_x=f_x, f_y=f_y, w_real=w_real,
            )
            pred_deg[i] = round(float(pitch), 4)
        except Exception:
            n_skip_pipeline += 1
            print("-" * 40)
            print(f"bug: inference pipeline skipped frame id {frame_id} due to error")
            traceback.print_exc() # print the error message

    df["pred_deg"] = pred_deg
    df.to_csv(output_csv, index=False)

    print(f"pipeline_skip={n_skip_pipeline}, total={n_total}")

if __name__ == "__main__":
    main()
