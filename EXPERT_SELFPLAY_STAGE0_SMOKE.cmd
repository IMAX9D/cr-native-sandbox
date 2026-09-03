@echo off
setlocal
pushd "%~dp0"
"D:\AI_data\runtime\venv\Scripts\python.exe" scripts\smoke_expert_selfplay_v1.py --device auto --output "D:\AI_data\cr-native-core\expert-selfplay-v1\stage0-smoke.json"
set CODE=%ERRORLEVEL%
popd
if not "%CODE%"=="0" pause
exit /b %CODE%
