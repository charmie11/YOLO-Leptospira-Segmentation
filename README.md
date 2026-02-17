# YOLO-Leptospira-Segmentation: Deep Learning-based Segmentation for Leptospira in Microscopic Images

This repository provides an automated pipeline to extract and segment Leptospira using YOLOv11-seg, designed to assist in Microscopic Agglutination Test (MAT) analysis.

## How to get the source code

You can obtain the source code by cloning the repository using Git or by downloading it as a ZIP file.

### Option 1: Using Git (Recommended)

Cloning the repository is recommended as it allows you to easily update the code to the latest version.

```bash
git clone https://github.com/charmie11/YOLO-Leptospira-Segmentation.git
cd YOLO-Leptospira-Segmentation
```

### Option 2: Downloading ZIP

If you are not familiar with Git, you can download the source code as a ZIP file.

1. Click the green "Code" button at the top of this repository page.
1. Select "Download ZIP".
1. Extract the downloaded ZIP file to your local workspace.

[!NOTE] If you download the ZIP file, please ensure you keep the directory structure intact, as the script relies on relative paths for data processing.

## How to setup python environment

### Requirements

- Python: 3.11 (Tested and confirmed)
- OS: Ubuntu (Recommended with NVIDIA GPU), Windows, or macOS
- Hardware:
  - GPU: NVIDIA GPU with CUDA support is highly recommended for training and fast inference (e.g., RTX A6000).
  - CPU: Supported for inference and small-scale testing.

### Installation

1. **Create a virtual environment (Recommended)**

   It is recommended to use a virtual environment to avoid conflicts with other projects.
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies

   The installation command for PyTorch differs depending on whether you are using a GPU or CPU.

   For GPU Users (CUDA 12.x)
   If you have an NVIDIA GPU, use the following command to install the CUDA-enabled version of PyTorch:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

   For CPU Users (or macOS)
   If you do not have a compatible GPU, install the standard versions:
  
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The training and evaluation scripts expect the following directory structure.
Ensuring this structure is maintained is crucial for the YOLO framework to correctly associate images with their corresponding labels.

```bash
.
├── main_split_data.py       # Main execution script
└── data/                    # Generated automatically by the script
    ├── data.yaml            # Configuration file for YOLO (Path, Classes, etc.)
    ├── images/              # Image files for each split
    │   ├── train/           # Training images
    │   ├── val/             # Validation images
    │   └── test/            # Test images
    └── labels/              # Corresponding annotation files (YOLO format)
        ├── train/           # Training labels
        ├── val/             # Validation labels
        └── test/            # Test labels
```

### `data.yaml`

The `data.yaml` file defines the dataset paths and class information.
It should look like this:

```txt
# Path settings
path: /path/to/project/data  # Root directory of the dataset
train: images/train          # Relative path to training images
val: images/val              # Relative path to validation images
test: images/test            # Relative path to test images (optional)

# Classes
names:
  0: leptospira
  1: clot
  2: outlier
```

[!TIP]

- The path variable should be updated to the absolute path of your project's data folder on your local machine.
- If you are only interested in detecting Leptospira, you can modify the names list accordingly.

### Optiton 1: Use CVAT

This is our recommended workflow, utilizing the provided script to automate data organization.

- Annotation Tool: We use [CVAT](https://www.cvat.ai/) (Computer Vision Annotation Tool) for annotating Leptospira in microscopic images.
- Export Format: Please export your dataset in "Ultralytics YOLO Segmentation 1.0" format.
- `data.yaml`: The exported archive contains `data.yaml`, which defines the dataset paths and class information.
- Workflow: Place your exported dataset.zip in the project root and run the following command. The script will automatically extract, split, and organize the data into the data/ directory.

```bash
python main_split_dataset.py
```

### Option 2: Use other tools

You are free to use any other annotation tools (such as LabelStudio or Roboflow).
There are no specific restrictions on the tools used, provided that the final dataset adheres to the **Expected Directory Structure** and **YOLO segmentation format** (normalized coordinates and class indices) described above.

## YOLO: Training

The training script initializes the YOLOv8-seg model and starts the learning process using the dataset prepared in the previous step.

### Execution

Run the following command to start training:

```bash
python main_train_yolo.py
```

### Outputs

By default, the training results are saved in the `PROJECT_DIR/runs/segment/train` directory.
This directory includes the following data:

- `weights/best.pt`: The model weights that achieved the best performance on the validation set. This file is required for the testing phase.
- `weights/last.pt`: The weights from the final epoch.
- `results.csv`: A log of training and validation loss, as well as precision/recall metrics for each epoch.
- `confusion_matrix.png`: A matrix showing the model's classification performance.
- `results.png`: Visual plots of the training progress (loss and mAP curves).

## YOLO: Performance Evaluation and Data Prediction

After training, the model is applied to the unseen test set in two distinct steps:

1. scientific validation of the model's accuracy
1. extraction of quantitative data for research analysis

### 1. Model Evaluation (Scientific Validation)

This step calculates standard metrics (mAP, Precision, Recall) to prove the reliability of the trained model.

#### Execution

Run the evaluation script:

```bash
python main_evaluate_yolo.py
```

The script automatically locates the latest trained weights at `runs/segment/train/weights/best.pt` and evaluates the performance using the test split defined in `data.yaml`.

#### Results

The results are saved in the `runs/segment/val` directory:

- `confusion_matrix.png`: Visualizes how well the model distinguishes between classes (e.g., Leptospira vs. Clot).
- `PR_curve.png`, `F1_curve.png`: Standard performance graphs for publication.

### 2. Data Prediction (Quantitative Analysis)

This step extracts the actual counts of bacteria and aggregates them into a structured format for your biological analysis.

#### Execution

Run the evaluation script:

```bash
python main_predict_yolo.py
```

#### Results

The results are saved in the `runs/segment/predict` directory:

- image files: All the segmented leptospira, clot, and outlier is annotated. The annotation contains boundingbox, segmentation mask, object label, and confidence as shown in the figures below.
- text files: All the segmented objects in a test image are saved as a text file in `runs/segment/predict/labels`. The format is shown below.

```txt
<object-category-ID> <x1> <y1> <x2> <y2> ... <xn> <yn>
```


## Bibliography

If you use this software or the models in your research, please cite the following paper(s):

```bibtex
@article{NAKANO2024JMM,
  author = {Risa Nakano and Yuji Oyamada and Ryo Ozuru and Michinobu Yoshimura and Kenji Hiromatsu},
  title = {Objectification of evaluation criteria in microscopic agglutination test using deep learning},
  journal = {Journal of Microbiological Methods},
  volume = {222},
  pages = {106955},
  year = {2024},
  issn = {0167-7012},
  doi = {[https://doi.org/10.1016/j.mimet.2024.106955](https://doi.org/10.1016/j.mimet.2024.106955)},
  url = {[https://www.sciencedirect.com/science/article/pii/S0167701224000678](https://www.sciencedirect.com/science/article/pii/S0167701224000678)}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

### Key Terms

- Open Source: The full source code of this project is available and must remain open under the terms of AGPL-3.0.
- Derivative Works: Any modifications or larger works that include this code must also be licensed under AGPL-3.0 and made available to users.
- Commercial Use: As this project utilizes the [Ultralytics YOLO framework](https://github.com/ultralytics/ultralytics), commercial use is subject to [Ultralytics Licensing terms](https://www.ultralytics.com/legal/agpl-3-0-software-license). A separate commercial license from Ultralytics may be required for any commercial application.

For the full license text, please see the [LICENSE](LICENSE) file in this repository.
