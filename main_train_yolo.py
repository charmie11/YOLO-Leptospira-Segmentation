from pathlib import Path

import torch
from ultralytics import YOLO


def run_train(data_yaml_path):
    """
    学習と検証（学習中のVal）を実行する
    """

    # インスタンスセグメンテーション用の事前学習済みモデルをロード
    model = YOLO("yolov8n-seg.pt")

    selected_device = 0 if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if selected_device == 0 else "CPU"
    print(f"--- Using device: {selected_device} ({device_name}) ---")

    print(f"--- Starting Training---")
    model.train(
        data=str(Path(data_yaml_path).absolute()),
        epochs=100,           # 最大エポック数
        batch=64,
        patience=10,          # アーリーストッピング（10エポック改善がなければ終了）
        imgsz=512,            # 解像度
        device=selected_device,
        save=True,            # ログと重みを保存
        exist_ok=True,        # フォルダが既存でもエラーにしない
        plots=True            # 学習曲線のグラフを保存
    )

    actual_save_dir = Path(model.trainer.save_dir)
    best_pt_path = actual_save_dir / "weights" / "best.pt"

    return str(best_pt_path.absolute())


def main():

    # train AI
    yaml_path = "data/data.yaml"
    best_model_path = run_train(yaml_path)
    print(f"--- Best weight path: {best_model_path} ---")


if __name__ == '__main__':
    main()
