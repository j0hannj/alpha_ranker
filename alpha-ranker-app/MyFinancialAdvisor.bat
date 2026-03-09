@echo off
title MyFinancialAdvisor
cd /d "%~dp0"

if not exist "venv" (
    echo.
    echo   MyFinancialAdvisor - First Time Setup
    echo   ======================================
    echo.
    echo   Creating environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo   Installing dependencies...
    pip install -r requirements.txt
    echo.
    echo   Installation complete!
    echo.
)

call venv\Scripts\activate.bat
python src\app.py
