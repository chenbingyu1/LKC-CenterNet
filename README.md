# LKC-CenterNet

PyTorch implementation of **LKC-CenterNet: Large-Kernel Physical Constraints for Quasi-Circular Micro-Target Detection and Measurement**.

LKC-CenterNet is a CenterNet-based framework for detecting dense quasi-circular micro-targets with **center-only supervision** and estimating their radii without radius annotations. The implementation combines a large-kernel ResNet-50 backbone, decoupled center/radius training, and image-derived physical constraints for radius learning.

> This repository is research code. Review the dataset license, annotation rights, and trained-weight sharing permissions before publishing a fork or redistribution.

## Features

- Center-only target localization with heatmap and offset prediction.
- LK-ResNet50 backbone with LKConv blocks for wider receptive fields and global shape perception.
- Two-stage training:
  - **Center stage**: learns target center heatmaps and offsets.
  - **Radius stage**: freezes localization-related modules and optimizes the radius head.
- Unsupervised radius loss combining interior consistency, edge contrast, and a radius prior.
- Single-image, directory, video, FPS, heatmap, and ONNX-export inference modes.
- VOC-format dataset preparation, center-distance evaluation, and VOC-style mAP utilities.

## Repository Layout

```text
LKC-CenterNet/
|-- train.py                    # Two-stage training entry point
|-- predict.py                  # Inference entry point
|-- centernet.py                # Model wrapper and inference utilities
|-- eval_center_points.py       # Center-distance AP / precision / recall evaluation
|-- get_map.py                  # VOC-style mAP evaluation
|-- voc_annotation.py           # VOC split and training-list generation
|-- nets/
|   |-- centernet.py            # CenterNet model definitions
|   |-- resnet50.py             # ResNet-50 and LKConv backbone components
|   `-- centernet_training.py   # Optimization and loss helpers
|-- utils/
|   |-- dataloader.py           # VOC data loading and center-only targets
|   |-- utils_fit.py            # Training loop
|   `-- utils_radius.py         # Unsupervised physical radius loss
|-- model_data/                 # Class files, fonts, and optional pretrained weights
|-- logs/                       # Stage-1 and stage-2 checkpoints (ignored by Git)
|-- img/                        # Input images for directory inference
|-- img_out1/                   # Directory-inference outputs
`-- VOCdevkit/VOC2007/          # VOC dataset root (ignored by Git)
```

## Environment

The code is written for Python 3 and PyTorch. Python 3.10 with a CUDA-compatible PyTorch build is recommended.

```bash
conda create -n lkc-centernet python=3.10
conda activate lkc-centernet

# Install the PyTorch build appropriate for your CUDA version:
pip install torch torchvision

pip install numpy opencv-python pillow matplotlib scipy tqdm tensorboard thop torchsummary
```

Optional dependencies:

```bash
# Needed only for COCO-style mAP in get_map.py (map_mode = 4)
pip install pycocotools

# Needed only for ONNX export
pip install onnx onnx-simplifier
```

## Dataset Preparation

The training code expects Pascal VOC-style data:

```text
VOCdevkit/
`-- VOC2007/
    |-- Annotations/        # One XML file per image
    |-- JPEGImages/         # Input images
    `-- ImageSets/Main/
        |-- train.txt
        |-- val.txt
        `-- test.txt
```

Each XML object should contain a class name and a bounding box. When `center_only=True`, the loader converts each box into a center-point target during training.

### Important: regenerate training lists locally

Do **not** reuse the checked-in `2007_train.txt` or `2007_val.txt` paths directly. They may contain absolute paths from the original training machine. After placing your data in `VOCdevkit/`, edit `voc_annotation.py` as required and run:

```bash
python voc_annotation.py
```

This regenerates dataset splits and the `2007_train.txt` / `2007_val.txt` files for your local path.

### Class-file consistency

`model_data/voc_classes.txt` currently contains a legacy multi-class example list, while the included VOC annotations use the single class name `point`.

- For a single-class `point` experiment, create a class file containing only:

  ```text
  point
  ```

- Update `classes_path` in `train.py`, `centernet.py`, `predict.py`, `get_map.py`, and `eval_center_points.py` consistently.
- A checkpoint and its class file must match. If the class count changes, train a new checkpoint rather than loading an incompatible one.

## Training

All main training options are configured near the top of `train.py`; the script does not use command-line arguments.

Default settings include a 512 x 512 input, LKConv enabled, Adam optimization, a frozen-backbone warm-up, and a total of 150 epochs.

### Stage 1: center-only training

In `train.py`, set:

```python
model_path = ""
train_stage = "center"
```

Then run:

```bash
python train.py
```

Stage-1 checkpoints are saved under `logs/center_only/`.

### Stage 2: unsupervised radius learning

Select the best stage-1 checkpoint and update `train.py`:

```python
model_path = "logs/center_only/best_epoch_weights.pth"
train_stage = "radius"
```

Then run:

```bash
python train.py
```

For the radius stage, the code freezes the backbone and center-localization modules and optimizes the radius head using the physical radius loss. The default radius-prior range is 3.0 to 8.0 pixels; adjust `radius_min` and `radius_max` for your imaging scale.

## Inference

Configure `predict.py` before running it.

### Directory inference

The default mode is `dir_predict`. Place images in `img/`, configure the desired checkpoint mode, and run:

```bash
python predict.py
```

Key settings in `predict.py`:

```python
weight_mode = "radius_only"  # "center_only" or "radius_only"
mode = "dir_predict"         # predict, video, fps, dir_predict, heatmap, export_onnx
dir_origin_path = "img/"
dir_save_path = "img_out1/"
```

With `weight_mode = "radius_only"`, the script writes a circle-overlay image and a three-panel visualization containing detected centers, estimated radii, and the center heatmap.

### Single-image inference

Set:

```python
mode = "predict"
```

Then run `python predict.py` and enter an image path when prompted.

## Evaluation

### Center localization

`eval_center_points.py` evaluates center predictions using a center-distance threshold. Configure the checkpoint path, class file, VOC root, confidence threshold, and `dist_thresh` at the top of the script:

```bash
python eval_center_points.py
```

The script reports mean center distance, precision, recall, and distance-threshold AP.

### VOC-style mAP

Configure `get_map.py` and run:

```bash
python get_map.py
```

Set `map_mode = 0` to generate predictions and ground truth before computing VOC-style mAP. Evaluation artifacts are written to `map_out/`.

## Reproducibility Notes

- The supplied data split contains 704 training images, 201 validation images, and 102 test images.
- The code sets `seed = 11` in `train.py`.
- GPU execution is enabled by default. Set `Cuda = False` in `train.py` or `cuda = False` in the `CenterNet` wrapper configuration for CPU inference/training where supported.
- Ensure that `input_shape`, class definitions, annotation format, and checkpoint configuration agree before loading weights.

## License and Attribution

This repository includes code derived from the Bubbliiiing CenterNet implementation and retains the included MIT License. Keep the original license and attribution notices in derivative distributions.

## Citation

If you use this code, please cite the associated manuscript:

```text
LKC-CenterNet: Large-Kernel Physical Constraints for Quasi-Circular
Micro-Target Detection and Measurement.
```

The full bibliographic citation and persistent identifier will be added after publication.
