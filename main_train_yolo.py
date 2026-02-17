from pathlib import Path

from ultralytics import YOLO

from yolo_utils import get_device


def main():

    # train AI
    data_yaml_path = Path("data/data.yaml")

    # インスタンスセグメンテーション用の事前学習済みモデルをロード
    model = YOLO("yolov8n-seg.pt")
    selected_device = get_device()

    print("--- Starting Training---")
    imgsz = 512
    model.train(
        data=str(data_yaml_path.absolute()),
        epochs=100,           # 最大エポック数
        batch=64,
        patience=10,          # アーリーストッピング（10エポック改善がなければ終了）
        imgsz=imgsz,            # 解像度
        device=selected_device,
        exist_ok=True,        # フォルダが既存でもエラーにしない
    )


if __name__ == '__main__':
    main()
