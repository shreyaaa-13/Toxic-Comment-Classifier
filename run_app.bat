@echo off
echo ========================================
echo Starting Toxic Comment Classifier App
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>NUL
if errorlevel 1 (
    echo ERROR: Streamlit not found!
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\lstm_model.h5" (
    echo WARNING: Models not found!
    echo Please train models first: python src\train_models.py
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 0
)

echo Starting Streamlit app...
echo.
streamlit run app.py

pause
