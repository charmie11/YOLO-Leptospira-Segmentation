import os
import glob
import yaml

from torch.cuda import is_available, get_device_name


def get_device():
    selected_device = 0 if is_available() else 'cpu'
    device_name = get_device_name(0) if selected_device == 0 else "CPU"
    print(f"--- Using device: {selected_device} ({device_name}) ---")

    return selected_device


def get_training_result(train_dir_path):
    train_dirs = glob.glob(train_dir_path)
    if not train_dirs:
        raise FileNotFoundError("Training directory not found. Run train.py first.")
    latest_train_dir = max(train_dirs, key=os.path.getmtime)

    best_model_path = os.path.join(latest_train_dir, 'weights', 'best.pt')

    args_yaml_path = os.path.join(latest_train_dir, 'args.yaml')
    if not os.path.exists(args_yaml_path):
        raise FileNotFoundError("Training settings yaml file does not exist. Run train.py first.")
    with open(args_yaml_path, 'r') as f:
        train_args = yaml.safe_load(f)
        data_yaml_path = train_args['data']

    return best_model_path, data_yaml_path
