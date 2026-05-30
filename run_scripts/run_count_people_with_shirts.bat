@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" scripts\count_people_video.py --video entrance.mov --model yolov8s.pt --shirt-classifier models\shirt_classifier_best.pt --zones configs\counting_zones.json --output-dir Dataset\counting_output --output-video entrance_counted_with_shirts.mp4 --process-every 3 --imgsz 512 --device 0
pause
