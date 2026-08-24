@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_human_vs_ai.ps1"
if errorlevel 1 (
  echo.
  echo Human-vs-AI startup failed.
  pause
  exit /b 1
)
