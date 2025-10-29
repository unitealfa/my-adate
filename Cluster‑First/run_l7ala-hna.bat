@echo off
REM Run the main.py script from the Cluster-First project directory.
pushd "%~dp0"
python "%~dp0main.py"
popd
pause