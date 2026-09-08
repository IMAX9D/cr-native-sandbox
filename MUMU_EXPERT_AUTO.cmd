@echo off
setlocal
title CR Expert - MuMu Live Controller
echo MuMu Expert: waits for a controllable live battle; trophies may be affected.
echo Close this window or press Ctrl+C to stop the controller.
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\start_mumu_expert.ps1" %*
if errorlevel 1 (
  echo.
  echo MuMu expert controller stopped with an error.
  pause
  exit /b 1
)
endlocal
