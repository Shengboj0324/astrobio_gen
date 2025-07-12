@echo off
echo ================================================================
echo  Astrobiology Project - Windows GPU Setup
echo ================================================================

echo.
echo [1/6] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10 first.
    pause
    exit /b 1
)

echo.
echo [2/6] Creating virtual environment...
python -m venv astrobio_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [3/6] Activating virtual environment...
call astrobio_env\Scripts\activate.bat

echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/6] Installing PyTorch with CUDA support...
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

echo.
echo [6/6] Installing project dependencies...
pip install -r requirements_windows_gpu.txt
pip install torch_geometric torch_sparse

echo.
echo ================================================================
echo  Testing GPU Setup...
echo ================================================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo.
echo ================================================================
echo  Setup Complete! 
echo  Activate environment with: astrobio_env\Scripts\activate
echo ================================================================
pause 