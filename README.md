# Valo-YOLO

Object detection for Valorant using YOLOv8 models.

This project includes two Jupyter notebooks: one for training the model and another for testing the dataset.

## Installation & Requirements

- Python 3.8
- NVIDIA GPU (optional for faster training)
- [PyTorch](https://pytorch.org/get-started/locally/) (Required for both training and inference)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the project, run the Jupyter notebooks:

1. **Training**: This notebook demonstrates the training process using the Valorant dataset.
2. **Testing**: This notebook tests the trained YOLOv8 model on sample images or videos.

Both notebooks require a YOLOv8 model trained on the [HuggingFace Valorant Object Detection dataset](https://huggingface.co/datasets/keremberke/valorant-object-detection).

## Dataset

The dataset used for training is available on [HuggingFace](https://huggingface.co/datasets/keremberke/valorant-object-detection).

## Requirements Summary

- Python 3.8
- PyTorch
- YOLOv8 model files
- Jupyter Notebook
