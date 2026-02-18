import shutil
from pathlib import Path
import zipfile
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image


def is_valid_image(path):
    """Pillowで画像として開けるか確認"""
    try:
        with Image.open(path) as img:
            return True
    except:
        return False


def main():
    zip_path = "data.zip"
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

    extract_dir = Path(zip_path).stem
    extract_path = Path(extract_dir).absolute()

    # 1. ZIP展開
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # 2. フォルダのリネーム (Train -> train)
    for parent in ["images", "labels"]:
        old_dir = extract_path / parent / "Train"
        new_dir = extract_path / parent / "train"
        if old_dir.exists():
            shutil.move(str(old_dir), str(new_dir))

    # 3. 画像以外の不要ファイルを削除 & 有効な画像リスト作成
    train_img_dir = extract_path / "images" / "train"
    all_images = []

    for f in list(train_img_dir.iterdir()):  # 削除しながらループするので list化
        if f.is_file():
            if is_valid_image(f):
                all_images.append(f)
            else:
                f.unlink()  # .DS_Store や壊れた画像をここで削除

    all_images.sort()
    image_names = [f.name for f in all_images]
    print(f"Valid images found: {len(image_names)}")

    # 4. データ分割（以下、移動処理）
    train_names, test_names = train_test_split(image_names, test_size=(1 - train_ratio), random_state=42)
    val_names, test_names = train_test_split(test_names, test_size=test_ratio / (val_ratio + test_ratio),
                                             random_state=42)

    split_map = {"train": train_names, "val": val_names, "test": test_names}

    for subset, names in split_map.items():
        (extract_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (extract_path / "labels" / subset).mkdir(parents=True, exist_ok=True)

        for name in tqdm(names, desc=f"Moving {subset}"):
            # images/train から各サブセットへ移動（※train同士なら実質何もしない）
            src_img = extract_path / "images" / "train" / name
            dst_img = extract_path / "images" / subset / name
            if src_img.exists() and src_img != dst_img:
                shutil.move(src_img, dst_img)

            # 対応するラベルの移動
            lbl_name = Path(name).with_suffix(".txt").name
            src_lbl = extract_path / "labels" / "train" / lbl_name
            dst_lbl = extract_path / "labels" / subset / lbl_name
            if src_lbl.exists() and src_lbl != dst_lbl:
                shutil.move(src_lbl, dst_lbl)

    # 6. data.yaml の修正
    yaml_path = Path(extract_dir) / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # 不要なタグを削除し、ディレクトリ指定へ更新
        data_config.pop('Train', None)  # 大文字のTrainタグを削除
        data_config.pop('train', None)  # 既存のtrain設定があれば上書き用に削除

        data_config['path'] = str(extract_path)
        data_config['train'] = "images/train"  # ディレクトリを指定
        data_config['val'] = "images/val"  # ディレクトリを指定
        data_config['test'] = "images/test"  # ディレクトリを指定

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    print("Done!")


if __name__ == "__main__":
    main()
