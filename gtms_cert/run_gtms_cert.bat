@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "PYTHON=python"
set "SEEDS=21 22 23 24 25 26"
set "MAX_PROCS=6"
set "TARGET_GAP=1"
set "RUNS_DIR=runs"

pushd "%PROJECT_ROOT%" >nul 2>&1
if errorlevel 1 (
    echo Impossible d'acceder au projet depuis %PROJECT_ROOT%.
    pause
    exit /b 1
)

echo ================================================
echo   Lancement du solveur GTMS-Cert (execution multi-graines)
echo ================================================

echo [INFO] Creation du dossier de journaux "%RUNS_DIR%" si besoin.
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

set "CMD=%PYTHON% -m gtms_cert.parallel_supervisor --max-procs %MAX_PROCS% --target-gap %TARGET_GAP% --runs-dir %RUNS_DIR% --seeds %SEEDS%"

echo [INFO] Commande : %CMD%
%CMD%
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo [OK] Supervision terminee. Consultez les journaux dans %RUNS_DIR%.
) else (
    echo.
    echo [ERREUR] L'execution s'est terminee avec le code %EXIT_CODE%.
)

popd >nul 2>&1
pause
exit /b %EXIT_CODE%
