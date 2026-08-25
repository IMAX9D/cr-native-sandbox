@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_expert_training_v1.ps1" -Smoke
if errorlevel 1 (
  echo.
  echo Expert v1 smoke test failed.
  pause
  exit /b 1
)
echo Expert v1 smoke test passed.
pause

