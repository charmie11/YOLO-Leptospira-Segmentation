import zipfile
import yaml
import shutil
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split


def main():
    zip_path = "data.zip"

    base_path = Path(zip_path).stem
    extract_dir = Path(base_path)
    yaml_path = extract_dir / "data.yaml"

    if extract_dir.exists() and yaml_path.exists():
        print(f"ディレクトリ '{extract_dir}' は既に存在します。分割処理をスキップします。")
        return yaml_path

    # 1. ZIPの展開
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 2. フォルダ・ファイル名の小文字化と正規化
    # Train.txt -> train.txt
    old_train_txt = extract_dir / "Train.txt"
    new_train_txt = extract_dir / "train.txt"
    if old_train_txt.exists():
        old_train_txt.rename(new_train_txt)

    # images/Train -> images/train, labels/Train -> labels/train
    for parent in ["images", "labels"]:
        old_dir = extract_dir / parent / "Train"
        new_dir = extract_dir / parent / "train"
        if old_dir.exists():
            if new_dir.exists():
                # 中身を移動して古い方を削除 (Errno 39 対策)
                for file in old_dir.iterdir():
                    shutil.move(str(file), str(new_dir / file.name))
                old_dir.rmdir()
            else:
                old_dir.rename(new_dir)

    if not new_train_txt.exists():
        raise FileNotFoundError(f"{new_train_txt} が見つかりません。")

    # 3. リストの読み込みと分割
    with open(new_train_txt, 'r') as f:
        # 読み込み時に Train -> train 置換
        all_image_paths = [line.strip().replace("/Train/", "/train/") for line in f if line.strip()]

    train_list, temp_list = train_test_split(all_image_paths, train_size=train_ratio, random_state=42)
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_list, test_list = train_test_split(temp_list, train_size=val_relative_ratio, random_state=42)

    subset_map = {"train": train_list, "val": val_list, "test": test_list}

    # 4. テキスト作成とYAML更新
    yaml_updates = {}
    for subset, paths in subset_map.items():
        list_file_name = f"{subset}.txt"
        list_file_abs_path = (extract_dir / list_file_name).absolute()
        with open(extract_dir / list_file_name, 'w') as f:
            # images/train/... -> images/subset/... に書き換え
            updated_lines = [p.replace("/train/", f"/{subset}/") for p in paths]
            f.write("\n".join(updated_lines) + "\n")
        yaml_updates[subset] = str(list_file_abs_path)

    # data.yaml の修正
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        data_config['path'] = str(extract_dir.absolute())
        data_config.pop('Train', None)
        for key, value in yaml_updates.items():
            data_config[key] = value

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

    # 5. ファイルの実体移動 (パスの二重重なり対策)
    print("\n--- Moving files to subdirectories ---")
    for subset in ["val", "test"]:
        if not subset_map[subset]: continue

        (extract_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (extract_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

        for img_rel_str in tqdm(subset_map[subset], desc=f"Processing {subset}"):
            # images フォルダ以降のパスを抽出してパスの重複を回避
            p = Path(img_rel_str)
            try:
                idx = p.parts.index("images")
                rel_path = Path(*p.parts[idx:])
            except ValueError:
                rel_path = p

            img_src = extract_dir / rel_path
            lbl_src = (extract_dir / str(rel_path).replace("images/", "labels/")).with_suffix(".txt")

            img_dst = extract_dir / "images" / subset / img_src.name
            lbl_dst = extract_dir / "labels" / subset / lbl_src.name

            if img_src.exists():
                shutil.move(str(img_src), str(img_dst))
            if lbl_src.exists():
                shutil.move(str(lbl_src), str(lbl_dst))

    print(f"YAML path: {yaml_path}")


if __name__ == '__main__':
    main()
