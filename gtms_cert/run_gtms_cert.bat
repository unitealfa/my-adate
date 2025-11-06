@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "EXIT_CODE=0"
pushd "%PROJECT_ROOT%" >nul

echo ================================================
echo   Lancement du solveur GTMS-Cert (instance generee dynamiquement)
echo ================================================

echo Entrez le nombre de camions disponibles :
set /p TRUCKS=

if "!TRUCKS!"=="" (
    echo Vous devez renseigner un nombre de camions.
    set "EXIT_CODE=1"
    goto END
)

set "NONNUM="
for /f "delims=0123456789" %%A in ("!TRUCKS!") do set "NONNUM=%%A"
if defined NONNUM (
    echo Merci d'entrer un nombre entier valide.
    set "EXIT_CODE=1"
    goto END
)

if !TRUCKS! LEQ 0 (
    echo Le nombre de camions doit etre strictement positif.
    set "EXIT_CODE=1"
    goto END
)

echo Entrez le nombre de clients pour le test (>= nombre de camions) :
set /p CLIENTS=

if "!CLIENTS!"=="" (
    echo Vous devez renseigner un nombre de clients.
    set "EXIT_CODE=1"
    goto END
)

set "NONNUM="
for /f "delims=0123456789" %%A in ("!CLIENTS!") do set "NONNUM=%%A"
if defined NONNUM (
    echo Merci d'entrer un nombre entier valide pour les clients.
    set "EXIT_CODE=1"
    goto END
)

if !CLIENTS! LSS !TRUCKS! (
    echo Le nombre de clients doit etre superieur ou egal au nombre de camions.
    set "EXIT_CODE=1"
    goto END
)

set "SEED_DEFAULT=42"
echo Entrez la graine aleatoire (defaut !SEED_DEFAULT!) :
set /p SEED=

if "!SEED!"=="" set "SEED=!SEED_DEFAULT!"

set "NONNUM="
for /f "delims=0123456789-" %%A in ("!SEED!") do set "NONNUM=%%A"
if defined NONNUM (
    echo Merci d'entrer un nombre entier valide pour la graine aleatoire.
    set "EXIT_CODE=1"
    goto END
)

python -m gtms_cert.run_with_custom_trucks --trucks !TRUCKS! --clients !CLIENTS! --seed !SEED! --lb-iters 0 --show --no-save

if errorlevel 1 (
    echo Echec de l'execution du solveur.
    set "EXIT_CODE=1"
    goto END
)

echo Execution terminee. La visualisation doit etre affichee ci-dessus.

:END
popd >nul
pause
exit /b %EXIT_CODE%
