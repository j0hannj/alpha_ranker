@echo off
title MyFinancialAdvisor - Build EXE
echo.
echo   ============================================
echo     MyFinancialAdvisor - Building Executable
echo   ============================================
echo.

cd /d "%~dp0"

:: Activate venv
if not exist "venv" (
    echo [!] Run INSTALLER.bat first to create the venv.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

:: Install PyInstaller if needed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [1/3] Installing PyInstaller...
    pip install pyinstaller
) else (
    echo [1/3] PyInstaller already installed.
)

echo [2/3] Building MyFinancialAdvisor.exe ...
echo         This takes 2-5 minutes on first build.
echo.

pyinstaller ^
    --name "MyFinancialAdvisor" ^
    --icon "icon.ico" ^
    --windowed ^
    --noconfirm ^
    --add-data "src;src" ^
    --add-data "icon.ico;." ^
    --add-data "icon.png;." ^
    --add-data "splash.png;." ^
    --hidden-import "customtkinter" ^
    --hidden-import "lightgbm" ^
    --hidden-import "xgboost" ^
    --hidden-import "sklearn" ^
    --hidden-import "sklearn.ensemble" ^
    --hidden-import "sklearn.linear_model" ^
    --hidden-import "scipy" ^
    --hidden-import "scipy.stats" ^
    --hidden-import "yfinance" ^
    --hidden-import "matplotlib" ^
    --hidden-import "matplotlib.backends.backend_tkagg" ^
    --hidden-import "PIL" ^
    --hidden-import "fredapi" ^
    --hidden-import "lxml" ^
    --collect-all "customtkinter" ^
    launcher.py

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo [3/3] Creating desktop shortcut...

:: Copy resources next to exe
copy /Y icon.ico "dist\MyFinancialAdvisor\" >nul
copy /Y icon.png "dist\MyFinancialAdvisor\" >nul
copy /Y splash.png "dist\MyFinancialAdvisor\" >nul
if not exist "dist\MyFinancialAdvisor\db" mkdir "dist\MyFinancialAdvisor\db"
if exist "db\portfolio.db" copy /Y "db\portfolio.db" "dist\MyFinancialAdvisor\db\" >nul
if exist ".env" copy /Y ".env" "dist\MyFinancialAdvisor\" >nul

:: Create desktop shortcut
powershell -Command ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$s = $ws.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'MyFinancialAdvisor.lnk'));" ^
    "$s.TargetPath = '%CD%\dist\MyFinancialAdvisor\MyFinancialAdvisor.exe';" ^
    "$s.WorkingDirectory = '%CD%\dist\MyFinancialAdvisor';" ^
    "$s.IconLocation = '%CD%\dist\MyFinancialAdvisor\icon.ico';" ^
    "$s.Description = 'MyFinancialAdvisor - Quantitative Investment Terminal';" ^
    "$s.Save()"

echo.
echo   ============================================
echo     BUILD COMPLETE!
echo   ============================================
echo.
echo   EXE location:  dist\MyFinancialAdvisor\MyFinancialAdvisor.exe
echo   Shortcut:      Desktop\MyFinancialAdvisor
echo.
echo   Double-click the shortcut to launch!
echo.
pause
