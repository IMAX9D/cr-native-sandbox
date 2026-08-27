@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=D:\AI_data\runtime\venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python.exe"
"%PYTHON%" "%~dp0scripts\native_generation_progress_gui.py"
if errorlevel 1 pause
