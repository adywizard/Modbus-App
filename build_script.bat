@echo off
setlocal

REM Ensure we’re using Python 3.13 and have deps installed
python -V
IF ERRORLEVEL 1 (
  echo Python not found.
  exit /b 1
)

REM Compile with Nuitka (standalone, no console, MSVC)
python -m nuitka --standalone --enable-plugin=tk-inter --include-data-files=modbus_settings.json=modbus_settings.json --windows-console-mode=disable --msvc=latest --output-dir=build --windows-icon-from-ico=app.ico --output-filename=ModbusTool.exe modbus_app.py

IF ERRORLEVEL 1 (
  echo Build failed.
  exit /b 1
)

echo Build OK. Find your app under build\modbus_app.dist\
endlocal