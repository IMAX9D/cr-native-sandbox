@echo off
setlocal
pushd "%~dp0"
if defined CR_EXPERT_PYTHON (
    start "" "%CR_EXPERT_PYTHON%" -m native_core.mumu_live_monitor
) else if exist "D:\AI_data\runtime\venv\Scripts\pythonw.exe" (
    start "" "D:\AI_data\runtime\venv\Scripts\pythonw.exe" -m native_core.mumu_live_monitor
) else (
    start "" pythonw -m native_core.mumu_live_monitor
)
popd
endlocal
