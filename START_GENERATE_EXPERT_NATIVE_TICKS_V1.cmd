@echo off
setlocal
set "PYTHON_EXE=D:\AI_data\runtime\venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python.exe"
pushd "%~dp0"

echo Ensuring four headless native libg workers are ready...
"%PYTHON_EXE%" -m native_core.worker start --workers 4 --transport direct
if errorlevel 1 goto :failed

echo Starting or resuming the authoritative native Tick dataset...
"%PYTHON_EXE%" "%~dp0scripts\generate_expert_native_ticks.py" run %*
if errorlevel 1 goto :failed

echo.
echo Native Tick dataset generation completed.
popd
exit /b 0

:failed
echo.
echo Native Tick dataset generation stopped. Existing work is safe to resume.
popd
pause
exit /b 1
