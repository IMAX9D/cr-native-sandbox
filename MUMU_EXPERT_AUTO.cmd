@echo off
setlocal
title CR Expert - MuMu Auto Friendly Battle
echo MuMu Expert: waiting for a friendly battle, then the AI takes control automatically.
echo Close this window or press Ctrl+C to stop the controller.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_mumu_expert.ps1"
if errorlevel 1 (
  echo.
  echo MuMu expert controller stopped with an error.
  pause
  exit /b 1
)
endlocal
