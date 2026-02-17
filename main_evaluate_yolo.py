from pathlib import Path
import glob
import os
import yaml

import pandas as pd
from ultralytics import YOLO


def save_results(results):
    # 保存ディレクトリの取得
    save_dir = results.save_dir

    # 1. クラスごとの指標を抽出
    # results.names はクラス名の辞書、results.map50 はクラスごとのmAP50（リスト）
    summary_data = []
    for i, name in results.names.items():
        summary_data.append({
            "class_id": i,
            "class_name": name,
            "precision": results.seg.p[i],
            "recall": results.seg.r[i],
            "mAP50": results.maps[i],  # クラスごとのmAP50
        })

    # 2. 全体平均（mean）の指標を追加
    summary_data.append({
        "class_id": "All",
        "class_name": "mean",
        "precision": results.results_dict["metrics/precision(M)"],
        "recall": results.results_dict["metrics/recall(M)"],
        "mAP50": results.results_dict["metrics/mAP50(M)"],
    })

    # 3. CSVとして保存
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(save_dir, 'evaluation_summary.csv')
    df.to_csv(csv_path, index=False)

    print(f"Summary metrics saved to: {csv_path}")


def main():
    # 1. 最新の学習ディレクトリを特定
    train_dirs = glob.glob(os.path.join('runs', 'segment', 'train*'))
    if not train_dirs:
        raise FileNotFoundError("Training directory not found. Run train.py first.")

    latest_train_dir = max(train_dirs, key=os.path.getmtime)
    best_model_path = os.path.join(latest_train_dir, 'weights', 'best.pt')
    model = YOLO(best_model_path)

    # 2. 学習時の設定から data.yaml のパスを取得
    args_yaml_path = os.path.join(latest_train_dir, 'args.yaml')
    with open(args_yaml_path, 'r') as f:
        train_args = yaml.safe_load(f)
        data_yaml_path = train_args['data']

    # 3. 評価と予測の実行
    results = model.val(data=data_yaml_path, split='test')
    save_results(results)

    # 4.
    test_image_dir = Path(data_yaml_path).parent / "images" / "test"
    model.predict(
        source=str(test_image_dir.absolute()),
        save=True,
        save_txt=True,
        imgsz=512,
        conf=0.25,
        exist_ok=True
    )


if __name__ == '__main__':
    main()
