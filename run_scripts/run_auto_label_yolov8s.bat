@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" scripts\auto_label_people_for_labelme.py --images Dataset\image_capture --output Dataset\labeledandimg --model yolov8s.pt --overwrite
pause
