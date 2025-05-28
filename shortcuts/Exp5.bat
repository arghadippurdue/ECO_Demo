@echo off
REM Activate Conda Environment
echo Activating Conda environment 'eco_demo'...
call conda activate eco_demo

REM Check if conda activation was successful
if %ERRORLEVEL% neq 0 (
    echo Failed to activate Conda environment. Please ensure 'eco_demo' exists.
    pause
    exit /b %ERRORLEVEL%
)

REM Change to the script directory
echo Changing directory to C:\Users\intel\ECO\ECO_Demo...
cd /D C:\Users\intel\ECO\ECO_Demo

REM Check if directory change was successful
if %ERRORLEVEL% neq 0 (
    echo Failed to change directory. Please ensure the path is correct.
    pause
    exit /b %ERRORLEVEL%
)

REM Run the Python script
echo ..............Starting Experiment 5...........
python run.py --mode realtime --framework ov --backbone mit_b2 --depth --noise 99 --experiment 5 --reportpower --device NPU

REM Pause to see output
echo Script execution finished.
pause
