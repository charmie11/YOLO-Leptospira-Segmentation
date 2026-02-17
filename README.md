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

- file structure

## Train YOLO

## Test YOLO

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
