# main.py
from utils.env_setup import setup_env
setup_env()

from utils.inference_road_lane_segmentation import main

if __name__ == "__main__":
    main()