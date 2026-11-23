@echo off
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :error
)
call .venv\Scripts\activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt || goto :error
python main.py %*
goto :eof
:error
echo.
echo Setup failed. Make sure Python 3.10-3.12 is installed and on PATH.
pause
