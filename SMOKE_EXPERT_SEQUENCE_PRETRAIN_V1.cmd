@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_expert_sequence_pretraining_v1.ps1" -Smoke -MinimumBattles 0
if errorlevel 1 (
  echo.
  echo Expert sequence smoke failed. Review the error above.
  pause
  exit /b 1
)
endlocal

