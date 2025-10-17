#!/bin/bash

echo "=== Fuzzy Classifier Practice Environment Setup ==="

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed. Please install Python first."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Ask if user wants to create virtual environment
read -p "Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    echo "Upgrading pip..."
    pip install --upgrade pip
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python dependencies"
    exit 1
fi

echo "=== Setup completed successfully! ==="
echo "You can now start working on the Fuzzy Classifier Practice project."
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "To activate the virtual environment: source venv/bin/activate"
fi
echo "Open this project in VSCode to get started."
echo "Make sure you have the Python extension installed in VSCode."
