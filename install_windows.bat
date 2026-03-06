@echo off
setlocal

REM ASCII-only wrapper to avoid cmd.exe garbling UTF-8/BOM batch files in some locales.
REM Run the PowerShell installer that ships with this repo.

set "SCRIPT_DIR=%~dp0"
REM Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\install_windows.ps1"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo [ERROR] install_windows.ps1 failed with exit code %EC%.
  echo.
)
exit /b %EC%

