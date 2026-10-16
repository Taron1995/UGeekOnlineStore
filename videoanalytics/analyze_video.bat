@echo off
rem Drag a video file (or a folder with videos) onto this file: the program does the rest and opens the report.
rem Double-click without a file: opens the window where you choose the videos.
setlocal
cd /d "%~dp0"
set "PY=python"
set "PYW=pythonw"
rem the virtual environment PyCharm creates for the project (.venv or venv), if there is one
if exist "venv\Scripts\python.exe" (
  set "PY=venv\Scripts\python.exe"
  set "PYW=venv\Scripts\pythonw.exe"
)
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
  set "PYW=.venv\Scripts\pythonw.exe"
)
if "%~1"=="" (
  start "" "%PYW%" run_gui.pyw
  exit /b
)
"%PY%" -m videoanalytics analyze %*
echo.
pause
