@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Determine repository root based on script location
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul

set "PYTHON_CMD=python"
set "TARGET_GAP=1.0"
set "MAX_PROCS=6"
set "TRUCKS=5"
set "CLIENTS=200"
set "RUNS_DIR=%CD%\runs"
set "SEEDS=21 22 23 24 25 26"

if not exist "%RUNS_DIR%" (
    mkdir "%RUNS_DIR%"
)

echo [INFO] Lancement du superviseur GTMS-Cert...
"%PYTHON_CMD%" -m gtms_cert.parallel_supervisor ^
    --seeds %SEEDS% ^
    --max-procs %MAX_PROCS% ^
    --target-gap %TARGET_GAP% ^
    --trucks %TRUCKS% ^
    --clients %CLIENTS% ^
    --runs-dir "%RUNS_DIR%" ^
    --solver-cmd "%PYTHON_CMD% -m gtms_cert.run_with_custom_trucks" ^
    --solver-extra --no-save --lb-iters 0

set "EXIT_CODE=%ERRORLEVEL%"
if %EXIT_CODE% NEQ 0 (
    echo [ERREUR] Le superviseur s'est termine avec le code %EXIT_CODE%.
) else (
    echo [INFO] Supervision terminee. Consultez les journaux dans %RUNS_DIR%.
)

echo.
echo Appuyez sur une touche pour fermer cette fenetre...
pause >nul

popd >nul
endlocal
