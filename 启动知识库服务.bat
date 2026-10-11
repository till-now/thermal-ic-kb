@echo off
title KB Server - http://192.168.254.57:8000/
cd /d "%~dp0"
set PY=C:\Users\田浩\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe

if not exist "%PY%" (
  echo [ERROR] Python not found:
  echo   %PY%
  echo Edit this file and fix the PY path.
  pause
  exit /b 1
)

echo ============================================================
echo   Thermal I&C Standards Knowledge Base
echo ============================================================
echo   Local  : http://localhost:8000/
echo   Phone  : http://192.168.254.57:8000/   (same WiFi)
echo.
echo   Close this window to stop the server.
echo ============================================================
echo.

start "" http://localhost:8000/
"%PY%" -m http.server 8000 --bind 0.0.0.0 --directory "%~dp0."

echo.
echo Server stopped.
pause
