@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "EXIT_CODE=0"
pushd "%SCRIPT_DIR%" >nul

echo ================================================
echo   Lancement du solveur GTMS-Cert (jeu de test 200 clients)
echo ================================================

set /p TRUCKS=Entrez le nombre de camions disponibles :

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

python -m gtms_cert.run_with_custom_trucks --trucks !TRUCKS! --lb-iters 0 --output "%SCRIPT_DIR%tests\solution_200_clients.json"
if errorlevel 1 (
    echo Echec de l'execution du solveur.
    set "EXIT_CODE=1"
    goto END
)

echo Execution terminee. Consultez le fichier tests\solution_200_clients.json pour les resultats.

:END
popd >nul
pause
exit /b %EXIT_CODE%