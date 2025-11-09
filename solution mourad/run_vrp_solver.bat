@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

rem Prefer virtualenv Python at repo root if present
set "VENV_PY=%SCRIPT_DIR%..\venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" "%SCRIPT_DIR%vrp_solver.py" %*
    goto :eof
)

rem Try Python launcher (py)
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 "%SCRIPT_DIR%vrp_solver.py" %*
    goto :eof
)

rem Fallback to python on PATH
python "%SCRIPT_DIR%vrp_solver.py" %*

endlocal

