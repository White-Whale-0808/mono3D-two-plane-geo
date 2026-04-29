import pandas as pd
import yaml

with open("config/convert_metadata_to_gt.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dir  = config["io"]["output_dir"]

def get_interval_mae(dataframe, start_id, end_id):

    mask = (dataframe['frame_id'] >= start_id) & (dataframe['frame_id'] <= end_id)
    interval_data = dataframe[mask].dropna(subset=['abs_error'])
    
    mae = interval_data['abs_error'].mean()
    count = len(interval_data)
    return mae, count

def main():
    df = pd.read_csv(input_dir)

    df['pred_deg'] = pd.to_numeric(df['pred_deg'], errors='coerce')
    df['gt_pitch_deg'] = pd.to_numeric(df['gt_pitch_deg'], errors='coerce')
    df['abs_error'] = (df['gt_pitch_deg'] - df['pred_deg']).abs()

    mae_1, count_1 = get_interval_mae(df, 196, 410)
    mae_2, count_2 = get_interval_mae(df, 410, 500)

    print(f"Interval 196~410: MAE = {mae_1:.4f} (Count: {count_1})")
    print(f"Interval 410~500: MAE = {mae_2:.4f} (Count: {count_2})")

if __name__ == "__main__":
    main()
