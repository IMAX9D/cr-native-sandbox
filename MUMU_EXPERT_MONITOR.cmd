@echo off
setlocal
pushd "%~dp0"
start "" "D:\AI_data\runtime\venv\Scripts\pythonw.exe" -m native_core.mumu_live_monitor
popd
endlocal
