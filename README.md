# Forest Fire Detection using Deep Learning

A production-ready, well-documented repository for detecting forest fires from images using deep learning. This project started as a Jupyter notebook (`Forest_Fire_detection_Using_Deep_Learning.ipynb`) and is organized so you can train, evaluate, and deploy a lightweight image classifier (Fire vs. No Fire).

> 🔥 Goal: Early detection of wildfires from aerial/satellite or ground images to help trigger rapid response.

---

## Table of Contents
- [Key Features](#key-features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference (Prediction)](#inference-prediction)
- [Export & Deployment](#export--deployment)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Key Features
- **Binary classification**: `fire` vs `no_fire` images.
- **Notebook-first** workflow with a clear path to scripts for automation.
- **Transfer learning ready** (e.g., ResNet, MobileNet, EfficientNet) or custom CNN.
- **Configurable training** (image size, augmentations, batch size, optimizer, LR schedulers).
- **Reproducible experiments** with seeds and deterministic settings.
- **Lightweight deployment** options (ONNX / TorchScript / TF-Lite) and Streamlit demo.
- **Clear documentation** and modular structure.

---

## Demo
You can run a quick interactive demo (Streamlit) after training:

```bash
streamlit run app.py
```

The demo lets you upload an image and returns the predicted class and confidence.

---

## Project Structure

```
.
├── Forest_Fire_detection_Using_Deep_Learning.ipynb   # Main notebook
├── app.py                                            # Optional: Streamlit demo (after training)
├── src/
│   ├── data/
│   │   ├── datamodule.py                             # Dataloaders & augmentations
│   │   └── utils.py                                  # Dataset prep helpers
│   ├── models/
│   │   ├── cnn.py                                    # Simple CNN baseline
│   │   └── factory.py                                # Transfer learning model factory
│   ├── train.py                                      # Scripted training entrypoint
│   ├── eval.py                                       # Evaluation & metrics
│   └── infer.py                                      # Batch/one-off inference utilities
├── experiments/
│   ├── runs/                                         # Auto-saved logs, checkpoints
│   └── configs/                                      # YAML configs for experiments
├── data/
│   ├── raw/                                          # Place raw images here
│   ├── processed/                                    # Train/val/test split
│   └── samples/                                      # Few example images
├── requirements.txt
└── README.md
```

> Tip: If these files don't exist yet, use the notebook to export or copy-paste the stubs from this README to bootstrap the `src/` structure.

---

## Requirements
- Python 3.9+
- Recommended packages (feel free to adjust):
  - `torch` or `tensorflow` (choose one; examples below assume **PyTorch**)
  - `torchvision`
  - `numpy`, `pandas`
  - `opencv-python`
  - `Pillow`
  - `scikit-learn`
  - `matplotlib`
  - `tqdm`
  - `albumentations` (optional, for rich augmentations)
  - `streamlit` (optional, for the demo)
  - `onnx`, `onnxruntime` (optional, for deployment)

Install:
```bash
pip install -r requirements.txt
```

Example `requirements.txt` (PyTorch flavor):
```
torch>=2.2.0
torchvision>=0.17.0
numpy
pandas
opencv-python
Pillow
scikit-learn
matplotlib
tqdm
albumentations
streamlit
onnx
onnxruntime
```

---

## Quick Start

1. **Clone** this repository and create a virtual environment.
2. **Prepare data** in `data/raw/` (see [Dataset Preparation](#dataset-preparation)).
3. **Run the notebook** to verify the pipeline end-to-end.
4. (Optional) **Use scripts** in `src/` for repeatable training/evaluation.
5. (Optional) **Launch demo** with `streamlit run app.py`.

---

## Dataset Preparation

This project expects a simple folder layout:
```
data/raw/
├── fire/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
└── no_fire/
    ├── img_101.jpg
    ├── img_102.jpg
    └── ...
```

Then split into train/val/test:
```bash
python -m src.data.utils --raw_dir data/raw --out_dir data/processed --val 0.15 --test 0.15 --seed 42
```

> You can use any image source (UAV, CCTV, satellite). Ensure balanced classes and diverse environments (day/night, haze, smoke, different terrains).

---

## Training

From the notebook **or** via script:

```bash
python -m src.train \
  --data_dir data/processed \
  --arch resnet18 \
  --img_size 224 \
  --batch_size 32 \
  --epochs 20 \
  --lr 3e-4 \
  --optimizer adamw \
  --aug strong \
  --seed 42 \
  --out_dir experiments/runs/resnet18_baseline
```

Common flags:
- `--arch`: `resnet18|resnet50|mobilenet_v3_small|efficientnet_b0|cnn`
- `--aug`: `none|light|strong`
- `--img_size`: typical values `160|192|224|256`
- `--mixup`, `--cutmix`: optional regularization
- `--fp16`: enable mixed precision

Checkpoints, logs, and metrics are saved under `experiments/runs/...`.

---

## Evaluation

Evaluate a trained checkpoint:
```bash
python -m src.eval \
  --data_dir data/processed \
  --checkpoint experiments/runs/resnet18_baseline/best.ckpt \
  --threshold 0.5
```

Outputs include:
- Accuracy, Precision, Recall, F1
- ROC-AUC & PR-AUC
- Confusion matrix
- Classification report
- Per-class metrics & curves

Artifacts are saved to the run folder (e.g., `confusion_matrix.png`, `roc_curve.png`).

---

## Inference (Prediction)

Single image:
```bash
python -m src.infer \
  --checkpoint experiments/runs/resnet18_baseline/best.ckpt \
  --image path/to/image.jpg
```

Batch (folder):
```bash
python -m src.infer \
  --checkpoint experiments/runs/resnet18_baseline/best.ckpt \
  --images_dir path/to/folder \
  --out_csv predictions.csv
```

The Streamlit app (`app.py`) wraps the same inference utilities for an interactive UI.

---

## Export & Deployment

Export to ONNX for edge inference:
```bash
python -m src.infer \
  --checkpoint experiments/runs/resnet18_baseline/best.ckpt \
  --export_onnx model.onnx --img_size 224
```

Options:
- **ONNX Runtime** for CPU inference.
- **TorchScript** if you deploy within a PyTorch app.
- **TF-Lite** (if training with TensorFlow).
- **Docker**: package the inference API as a container.
- **Streamlit**: local desktop demo.

---

## Results

> Replace the table below with your latest experiment (numbers are placeholders).

| Model            | Img Size | Acc. | Precision | Recall | F1   | ROC-AUC |
|------------------|----------|------|-----------|--------|------|---------|
| ResNet-18 (base) | 224      | 0.96 | 0.95      | 0.96   | 0.96 | 0.98    |
| MobileNet-V3     | 224      | 0.95 | 0.94      | 0.95   | 0.95 | 0.98    |

Include visual examples in `data/samples/` and confusion matrix plots under `experiments/runs/...`.

---

## Reproducibility
- Set `--seed 42` (configurable) for NumPy/PyTorch and dataloader workers.
- Limit non-determinism by disabling cuDNN benchmarking where necessary.
- Log package versions and GPU info in each run folder.

---

## Roadmap
- [ ] Multi-class extension (smoke-only, flame, aftermath, fog/haze distinctions)
- [ ] Real-time video processing (OpenCV / RTSP) with temporal smoothing
- [ ] Active learning loop for hard negatives
- [ ] Semi-supervised training on unlabeled data
- [ ] Model distillation for mobile deployment
- [ ] Geo-referencing & alert system integration

---

## Troubleshooting
- **Overfitting**: increase augmentations, use dropout, try MixUp/CutMix, early stopping.
- **Class imbalance**: weighted loss, focal loss, oversampling, hard negative mining.
- **False positives (clouds/fog)**: enrich `no_fire` with diverse negatives; use color-invariant features.
- **Blurry/low-res images**: reduce `--img_size` sensitivity or apply super-resolution preprocessing.
- **Slow training**: enable `--fp16`, reduce batch size, swap to MobileNet/EfficientNet-lite.

---

## Contributing
Contributions are welcome! Please open an issue to discuss major changes and follow these steps:

1. Fork the repo
2. Create a feature branch (`feat/my-feature`)
3. Commit with conventional messages
4. Open a PR with a clear description and screenshots/metrics

---

## License
This project is released under the **MIT License** (feel free to change this in `LICENSE`).

---

## Citation
If you use this work, please cite:
```
@software{forest_fire_detection_2025,
  title        = {Forest Fire Detection using Deep Learning},
  author       = {Your Name},
  year         = {2025},
  url          = {https://github.com/your-username/your-repo}
}
```

---

## Acknowledgements
- Thanks to open-source contributors and datasets used for research and benchmarking.
- Built with ❤️ using Python and modern deep learning libraries.
