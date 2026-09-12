@echo off
cd /d "%~dp0"
if defined CR_MATCH_PYTHON (
  "%CR_MATCH_PYTHON%" -m training.league_dashboard %*
) else (
  python -m training.league_dashboard %*
)
pause
