@echo off
title MyFinancialAdvisor - Installation
echo.
echo   ========================================
echo     MyFinancialAdvisor - Installation
echo   ========================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo   Install Python 3.11+: https://python.org/downloads
    echo   Check "Add python.exe to PATH"
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist "venv" python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing dependencies...
pip install -r requirements.txt

echo [3/4] Creating desktop shortcut...
python create_shortcut.py

echo [4/4] Done!
echo.
echo   Double-click "MyFinancialAdvisor" on your desktop to launch.
echo   Or run MyFinancialAdvisor.bat from this folder.
echo.
pause
