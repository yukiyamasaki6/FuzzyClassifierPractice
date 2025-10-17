@echo off
echo === Fuzzy Classifier Practice Environment Setup ===

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo Using Python: 
python --version

REM Ask if user wants to create virtual environment
set /p create_venv="Create virtual environment? (recommended) [y/N]: "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo Upgrading pip...
    python -m pip install --upgrade pip
)

echo Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install Python dependencies
    pause
    exit /b 1
)

echo === Setup completed successfully! ===
echo You can now start working on the Fuzzy Classifier Practice project.
if /i "%create_venv%"=="y" (
    echo To activate the virtual environment: venv\Scripts\activate.bat
)
echo Open this project in VSCode to get started.
echo Make sure you have the Python extension installed in VSCode.
pause
