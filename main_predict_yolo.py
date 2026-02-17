import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from yolo_utils import get_device, get_training_result


def main():
    # 1. 最新の学習ディレクトリを特定
    best_model_path, data_yaml_path = get_training_result(os.path.join('runs', 'segment', 'train*'))
    model = YOLO(best_model_path)
    selected_device = get_device()

    # 4.
    test_image_dir = Path(data_yaml_path).parent / "images" / "test"
    imgsz = 512
    conf = 0.5
    model.predict(
        source=str(test_image_dir.absolute()),
        imgsz=imgsz,
        conf=conf,
        device=selected_device,
        save=True,
        save_txt=True,
    )


if __name__ == '__main__':
    main()
