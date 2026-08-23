@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_selfplay_v0_1.ps1"
if errorlevel 1 (
  echo.
  echo Training failed. See D:\AI_data\cr-native-core\selfplay-v0.1\supervisor-logs
  pause
  exit /b 1
)
