import pandas as pd
import matplotlib.pyplot as plt

def plot_frameId_and_pitch():
    file_path = 'inference_datasets/solid_line/up_hile/metadata.csv'
    df = pd.read_csv(file_path)

    filtered_df = df[(df['frame_id'] >= 190) & (df['frame_id'] <= 765)]

    plt.plot(filtered_df['frame_id'], filtered_df['pitch'])
    plt.xlabel('frame_id')
    plt.ylabel('pitch')

    plt.show()

def main():
    plot_frameId_and_pitch()

if __name__ == "__main__":
    main()