@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_expert_one_click_v1.ps1" %*
if errorlevel 1 (
  echo.
  echo Expert one-click pipeline stopped fail-closed. Existing progress is safe to resume.
  pause
  exit /b 1
)
echo.
echo Expert one-click pipeline completed successfully.
pause
