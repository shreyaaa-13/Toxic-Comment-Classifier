@echo off
echo ========================================
echo Training Toxic Comment Classifier Models
echo ========================================
echo.

REM Check if data exists
if not exist "data\train.csv" (
    echo WARNING: Dataset not found!
    echo.
    set /p generate="Generate sample dataset? (y/n): "
    if /i "%generate%"=="y" (
        echo Generating sample data...
        python src\generate_sample_data.py
        echo.
    ) else (
        echo Please download the Kaggle dataset or generate sample data.
        pause
        exit /b 1
    )
)

echo Starting model training...
echo This may take several minutes...
echo.

python src\train_models.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Next step: Run the web app with run_app.bat
echo.

pause
