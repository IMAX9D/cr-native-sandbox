@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_training_dashboard.ps1"
if errorlevel 1 (
  echo.
  echo Training dashboard startup failed.
  pause
  exit /b 1
)
