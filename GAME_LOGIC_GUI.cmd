@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_logic_gui.ps1"
if errorlevel 1 (
  echo.
  echo GUI startup failed.
  pause
  exit /b 1
)
