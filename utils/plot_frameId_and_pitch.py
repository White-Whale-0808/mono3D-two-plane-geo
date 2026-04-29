import pandas as pd
import matplotlib.pyplot as plt

def plot_frameId_and_pitch():
    file_path = "outputs/metadata_gt.csv"
    df = pd.read_csv(file_path)

    df_filtered = df[df["gt_pitch_deg"].notna() & df["pred_deg"].notna()]

    plt.plot(df_filtered['frame_id'], df_filtered["gt_pitch_deg"], color="blue")
    plt.plot(df_filtered['frame_id'], df_filtered["pred_deg"], color= "red")
    plt.xlabel('frame_id')
    plt.ylabel('pitch')

    plt.savefig("outputs/pitch_plot.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    plot_frameId_and_pitch()

if __name__ == "__main__":
    main()