@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" scripts\crop_person_bodies.py --detector yolov8s.pt --process-every 3 --imgsz 512 --save-every 12 --max-crops-per-track 10
pause
