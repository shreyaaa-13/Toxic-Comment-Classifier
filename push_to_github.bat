@echo off
echo ========================================
echo Push to GitHub
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please download and install Git from: https://git-scm.com/downloads
    pause
    exit /b 1
)

REM Check if already initialized
if not exist ".git" (
    echo Initializing Git repository...
    git init
    echo.
)

echo Adding files to Git...
git add .
echo.

echo Creating commit...
set /p commit_message="Enter commit message (or press Enter for default): "
if "%commit_message%"=="" set commit_message=Update: Toxic Comment Classifier

git commit -m "%commit_message%"
echo.

REM Check if remote exists
git remote -v | find "origin" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo FIRST TIME SETUP
    echo ========================================
    echo.
    echo Please enter your GitHub repository URL:
    echo Example: https://github.com/YOUR-USERNAME/Toxic-Comment-Classifier.git
    echo.
    set /p repo_url="Repository URL: "
    
    git remote add origin %repo_url%
    git branch -M main
    echo.
    echo Remote added successfully!
    echo.
)

echo Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo Done!
echo ========================================
echo.
echo Your project is now on GitHub!
echo.

pause
