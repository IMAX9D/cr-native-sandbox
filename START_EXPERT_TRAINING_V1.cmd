@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_expert_training_v1.ps1"
if errorlevel 1 (
  echo.
  echo Expert v1 startup failed. The launcher is fail-closed; inspect the error above.
  pause
  exit /b 1
)

