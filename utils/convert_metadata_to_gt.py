import yaml
from pathlib import Path
import pandas as pd

with open("config/convert_metadata_to_gt.yaml", "r") as f:
    config = yaml.safe_load(f)

slope_deg = config["slope_deg"]
input_dir  = config["io"]["input_dir"]
output_dir = config["io"]["output_dir"]

def convert(input_dir, output_dir, slope_deg):
    df = pd.read_csv(input_dir)

    df["gt_pitch_deg"] = (slope_deg - df["pitch"].astype(float)).round(4)
    df.to_csv(output_dir, index=False)

    return len(df)


def main():
    n = convert(input_dir, output_dir, slope_deg)
    print(f"Written {n} rows to {output_dir}")

if __name__ == "__main__":
    main()
