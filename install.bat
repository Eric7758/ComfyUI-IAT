@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set "REQ=%cd%\requirements.txt"
if not exist "%REQ%" (
  echo [IAT] requirements.txt not found, skipping.
  exit /b 0
)

REM Best effort: try to use ComfyUI's embedded python if present.
set "PYTHON_EXE=python"
if exist "..\..\python_embeded\python.exe" set "PYTHON_EXE=..\..\python_embeded\python.exe"
if exist "..\..\python\python.exe" set "PYTHON_EXE=..\..\python\python.exe"

echo [IAT] Using Python: %PYTHON_EXE%
%PYTHON_EXE% -m pip install -r "%REQ%"
if errorlevel 1 (
  echo [IAT] pip install failed.
  exit /b 1
)

echo [IAT] Done.
exit /b 0

