@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_training.ps1"
if errorlevel 1 (
  echo.
  echo Training failed. See D:\AI_data\cr-native-core\training\launcher-logs
  pause
  exit /b 1
)
