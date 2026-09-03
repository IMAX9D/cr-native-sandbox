@echo off
setlocal
echo Expert v1.2 - control step 157674 - 2.0x play rate - Queen/Hog deck
echo Close the existing local match window before starting this test.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_human_vs_ai.ps1" -Checkpoint "D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831\control-lr1e-4-step157674-fp16.pt" -Replay "%~dp0examples\queen-hog-control.json" -BattleSeed 2 -ExpertPlayRateScale 2.0
if errorlevel 1 (
  echo Expert 2.0x test startup failed.
  pause
  exit /b 1
)
endlocal
