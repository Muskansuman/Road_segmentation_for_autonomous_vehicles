# Deep Learning Image Segmentation System

Semantic and instance segmentation pipeline for road scenes, including off-road, Indian road, and low-light conditions — built for autonomous driving applications.

[![GitHub](https://img.shields.io/badge/GitHub-Muskansuman-blue)](https://github.com/Muskansuman/Road-Segmentation-for-Autonomous-Vehicles-)

## Overview

This project compares classical computer vision methods with deep learning approaches for image segmentation:

| Approach | Method | Use case |
|----------|--------|----------|
| Classical | K-Means, GMM | Unsupervised pixel clustering baseline |
| Semantic | UNet (PyTorch) | Pixel-level road vs. non-road classification |
| Instance | YOLOv8 | Object-level segmentation with masks |
| Instance | Detectron2 | Meta AI mask R-CNN style segmentation |

**Dataset:** Custom IITJ_RoadSeg dataset (~1000+ images) captured on campus, annotated with [Roboflow](https://roboflow.com/), available in PNG mask and COCO formats.

## Project Structure

```
.
├── GMM.py                          # K-Means & GMM classical segmentation
├── UNET_CustomDataset.py           # UNet training pipeline (PyTorch)
├── InstanceSegmentation.ipynb      # YOLOv8 instance segmentation
├── Detectron2_Segmentation_on_CustomData (1).ipynb
├── demo.gif                        # Roboflow annotation workflow
├── k_means_result.png
├── yolo_result.png
├── detectron_result.png
├── Results.gif
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Muskansuman/Road-Segmentation-for-Autonomous-Vehicles-.git
cd Road-Segmentation-for-Autonomous-Vehicles-

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA-capable GPU recommended for UNet and YOLOv8 training.

## Dataset Setup

Organize your data before training UNet:

```
data/
└── training/
    ├── images/     # Input road images (.png / .jpg)
    └── masks/      # Ground-truth segmentation masks
```

For YOLOv8, export your dataset from Roboflow in YOLOv8 format and update the path in `InstanceSegmentation.ipynb`.

## Usage

### 1. Classical Segmentation (K-Means / GMM)

```bash
python GMM.py --image road.png --clusters 3 --method gmm
```

Options:
- `--method kmeans` — K-Means clustering
- `--method gmm` — Gaussian Mixture Model

### 2. Semantic Segmentation (UNet)

```bash
python UNET_CustomDataset.py \
  --img-path data/training/images \
  --mask-path data/training/masks \
  --epochs 150 \
  --batch-size 32
```

The best model checkpoint is saved as `best_model_state.bin`.

### 3. Instance Segmentation (YOLOv8)

Open `InstanceSegmentation.ipynb` in Google Colab or Jupyter with GPU enabled.

Train:
```bash
yolo task=segment mode=train model=yolov8m-seg.pt data=path/to/data.yaml epochs=10 imgsz=640
```

Inference:
```bash
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt conf=0.25 source=path/to/test/images save=true
```

### 4. Instance Segmentation (Detectron2)

Open `Detectron2_Segmentation_on_CustomData (1).ipynb` and follow the notebook cells. Requires a GPU runtime.

## Results

### K-Means Clustering
<img src="k_means_result.png" width="300" alt="K-Means result">

### YOLOv8 Instance Segmentation — mAP50: 0.908
<img src="yolo_result.png" width="500" alt="YOLOv8 result">

### Detectron2 Instance Segmentation
<img src="detectron_result.png" width="350" alt="Detectron2 result">

### UNet Training
- Average validation IoU loss: **0.03**
- Architecture: UNet with IoU loss, Albumentations augmentation, Adam optimizer

### Annotation Workflow (Roboflow)
<img src="demo.gif" width="600" alt="Roboflow annotation demo">

## Key Features

- Custom dataset creation and annotation pipeline
- UNet implemented from scratch in PyTorch (encoder-decoder with skip connections)
- IoU loss, data augmentation, train/validation/test split
- Comparison of classical vs. deep learning segmentation
- Support for challenging scenes: off-road, low-light, Indian road conditions

## References

- [Roboflow — Dataset annotation](https://roboflow.com/)
- [KITTI Road Segmentation — PyTorch UNet](https://www.kaggle.com/code/hossamemamo/kitti-road-segmentation-pytorch-unet-from-scratch)

## Contributor

**[Muskan Suman](https://github.com/Muskansuman)** — Project owner & maintainer

## License

This project is for educational and portfolio purposes.
