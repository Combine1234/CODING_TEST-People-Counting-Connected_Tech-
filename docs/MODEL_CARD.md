# Model Card

## Person Detector

- Default model: `yolov8s.pt`
- Task: person detection
- Source: Ultralytics pretrained weights
- Usage: downloaded automatically by Ultralytics if missing

## Shirt Classifier

- Model file: `models/shirt_classifier_best.pt`
- Base architecture: YOLOv8n classification
- Classes:
  - `Superai_Shirt`
  - `Unknow_Shirt`
- Local training split:
  - Train images: 571
  - Validation images: 143
- Local validation:
  - top1 accuracy: 0.99301

## Limitations

- The validation set comes from the same video/camera domain, so real-world generalization is not guaranteed.
- Occlusion, motion blur, and partial body crops may reduce shirt classification quality.
- The classifier predicts the crop class; it does not segment the exact shirt area.
