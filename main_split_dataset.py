import shutil
from pathlib import Path
import zipfile

import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    zip_path = "data.zip"
    # 比率の設定
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    base_path = Path(zip_path).stem
    extract_dir = Path(base_path).absolute()
    yaml_path = extract_dir / "data.yaml"

    if extract_dir.exists() and yaml_path.exists():
        print(f"ディレクトリ '{extract_dir}' は既に存在します。")
        return yaml_path

    # 1. ZIPの展開
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 2. フォルダ名の正規化 (Train -> train)
    # フォルダベースで管理するため、 images/train, labels/train を起点にします
    for parent in ["images", "labels"]:
        old_dir = extract_dir / parent / "Train"
        new_dir = extract_dir / parent / "train"
        if old_dir.exists():
            # new_dir (train) が存在しない場合のみ rename を実行
            if not new_dir.exists():
                old_dir.rename(new_dir)
            else:
                # 万が一両方存在した場合は、安全のため中身を統合して古い方を消す
                for file in old_dir.iterdir():
                    shutil.move(str(file), str(new_dir / file.name))
                old_dir.rmdir()

    # 3. 不要なテキストファイルの削除
    # 旧仕様の Train.txt などがZIPに含まれていた場合に備えて削除
    for txt_file in ["Train.txt", "train.txt"]:
        target_txt = extract_dir / txt_file
        if target_txt.exists():
            target_txt.unlink()
            print(f"Removed legacy file: {txt_file}")

    # 4. ファイルリストの取得と分割（フォルダ内の実体ファイルからリストを作成）
    all_images = sorted(list((extract_dir / "images" / "train").glob("*")))
    all_image_names = [f.name for f in all_images]

    if not all_image_names:
        raise FileNotFoundError("images/train 内に画像が見つかりません。")

    train_names, temp_names = train_test_split(all_image_names, train_size=train_ratio, random_state=42)
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_names, test_names = train_test_split(temp_names, train_size=val_relative_ratio, random_state=42)

    subset_map = {"val": val_names, "test": test_names}

    # 5. ファイルの実体移動 (val, test 用に画像を振り分け)
    print("\n--- Moving files to subdirectories ---")
    for subset, names in subset_map.items():
        (extract_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (extract_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

        for name in tqdm(names, desc=f"Processing {subset}"):
            # 画像の移動
            img_src = extract_dir / "images" / "train" / name
            img_dst = extract_dir / "images" / subset / name
            if img_src.exists():
                shutil.move(str(img_src), str(img_dst))

            # ラベルの移動 (.txt)
            lbl_name = Path(name).with_suffix(".txt").name
            lbl_src = extract_dir / "labels" / "train" / lbl_name
            lbl_dst = extract_dir / "labels" / subset / lbl_name
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(lbl_dst))

    # 6. data.yaml の修正
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # 不要なタグを削除し、ディレクトリ指定へ更新
        data_config.pop('Train', None)  # 大文字のTrainタグを削除
        data_config.pop('train', None)  # 既存のtrain設定があれば上書き用に削除

        data_config['path'] = str(extract_dir)
        data_config['train'] = "images/train"  # ディレクトリを指定
        data_config['val'] = "images/val"  # ディレクトリを指定
        data_config['test'] = "images/test"  # ディレクトリを指定

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

    print(f"Processing complete. All configurations set to directory-based paths.")


if __name__ == '__main__':
    main()