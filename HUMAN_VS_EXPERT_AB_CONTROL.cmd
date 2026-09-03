@echo off
setlocal
echo Expert v1.2 - control experiment - step 157674 - Queen/Hog deck
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_human_vs_ai.ps1" -Checkpoint "D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831\control-lr1e-4-step157674-fp16.pt" -Replay "%~dp0examples\queen-hog-control.json" -BattleSeed 2
if errorlevel 1 (
  echo Expert experiment startup failed.
  pause
  exit /b 1
)
endlocal
