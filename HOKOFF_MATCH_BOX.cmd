@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_hokoff_match_box.ps1" %*
if errorlevel 1 (
  echo BC match box startup failed. See the error above.
  pause
  exit /b 1
)
