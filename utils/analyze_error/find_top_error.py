import pandas as pd
import yaml

with open("config/convert_metadata_to_gt.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dir  = config["io"]["output_dir"]
output_dir = "outputs/top_errors.csv"
top_k = 10
def main():
    df = pd.read_csv(input_dir)

    df['pred_deg'] = pd.to_numeric(df['pred_deg'], errors='coerce')
    df['gt_pitch_deg'] = pd.to_numeric(df['gt_pitch_deg'], errors='coerce')
    df['abs_error'] = (df['gt_pitch_deg'] - df['pred_deg']).abs()

    top_10_errors = df.nlargest(top_k, 'abs_error')
    top_10_errors.to_csv(output_dir, index=False)

if __name__ == "__main__":
    main()