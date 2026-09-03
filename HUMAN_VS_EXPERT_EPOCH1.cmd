@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_human_vs_ai.ps1" -Checkpoint "D:\AI_data\cr-native-core\expert-v1\downloaded\expert-v1.1-cloud-177m-b32\epoch-001-fp16.pt"
if errorlevel 1 (
  echo Human-vs-Expert Epoch 1 startup failed.
  pause
  exit /b 1
)
endlocal
