# Project Pipeline

## 1. Dataset Creation

The source video `entrance.mov` is sampled every 1 second. The extracted frames are used as the initial dataset for inspection, auto-labeling, and future detector improvement.

Script:

```bat
python scripts\capture_entrance_frames.py --video entrance.mov --output Dataset\image_capture --interval 1
```

## 2. Person Auto-Labeling

YOLO detects `person` objects on the extracted frames. The detected boxes are written as LabelMe JSON files, so the labels can be manually corrected before future training.

Script:

```bat
python scripts\auto_label_people_for_labelme.py --images Dataset\image_capture --output Dataset\labeledandimg --model yolov8s.pt
```

## 3. Shirt Crop Dataset

The video is tracked and person crops are exported. Full-body crops and torso crops are saved separately. Torso crops are suitable for shirt classification.

Script:

```bat
python scripts\crop_person_bodies.py --detector yolov8s.pt
```

## 4. Shirt Classification

The shirt classifier is trained from class folders:

```text
Superai_Shirt
Unknow_Shirt
```

The trained model is saved as:

```text
models\shirt_classifier_best.pt
```

## 5. Full Video Inference

The final pipeline combines:

- person detection
- tracking and lightweight ReID
- zone event counting
- shirt classification per tracked person
- annotated video export

Main script:

```bat
python scripts\count_people_video.py --video entrance.mov --model yolov8s.pt --shirt-classifier models\shirt_classifier_best.pt
```
