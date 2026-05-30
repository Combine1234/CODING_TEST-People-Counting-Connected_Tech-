@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" scripts\train_shirt_classifier.py --source Dataset\shirt_crop_dataset --dataset Dataset\shirt_cls_dataset --model yolov8n-cls.pt --epochs 25 --imgsz 224 --batch 32 --device 0
pause
