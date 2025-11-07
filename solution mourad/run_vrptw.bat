@echo off
setlocal

rem S'assure que l'on travaille depuis le dossier du script (gère les espaces).
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

rem Ajoute le dossier src au PYTHONPATH pour que le module vrptw soit trouvé.
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

rem Lance l'assistant en mode interactif par défaut (possibilité d'ajouter des arguments).
python -m vrptw.cli --interactive %*
set "EXIT_CODE=%ERRORLEVEL%"

rem Revient au dossier d'origine et informe en cas d'erreur.
popd >nul
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Le script s'est termine avec le code %EXIT_CODE%.
)

echo.
pause
endlocal & exit /b %EXIT_CODE%