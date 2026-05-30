@echo off
setlocal
cd /d "%~dp0\.."

set PYTHON_EXE=%CD%\.venv_gpu\Scripts\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

"%PYTHON_EXE%" scripts\capture_entrance_frames.py --video entrance.mov --output Dataset\image_capture --interval 1 --overwrite
pause
