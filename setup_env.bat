@ECHO OFF
REM Script to initialize a Python virtual environment and install dependencies on Windows.

SET VENV_NAME=cr
SET PYTHON_MAJOR_REQUIRED=3
SET PYTHON_MINOR_REQUIRED=12

ECHO --- Checking Python Version (Required: %PYTHON_MAJOR_REQUIRED%.%PYTHON_MINOR_REQUIRED%+) ---

REM Try to run a python command that exits with errorlevel 0 if version is OK, 1 otherwise.
python -c "import sys; sys.exit(0) if sys.version_info >= (%PYTHON_MAJOR_REQUIRED%, %PYTHON_MINOR_REQUIRED%) else sys.exit(1)"
IF ERRORLEVEL 1 (
    ECHO Error: Python %PYTHON_MAJOR_REQUIRED%.%PYTHON_MINOR_REQUIRED%+ is required.
    ECHO Please install Python %PYTHON_MAJOR_REQUIRED%.%PYTHON_MINOR_REQUIRED% or newer and ensure it's in your PATH.
    ECHO You might need to use the 'py' launcher e.g., 'py -%PYTHON_MAJOR_REQUIRED%.%PYTHON_MINOR_REQUIRED% -m venv %VENV_NAME%' if 'python' is not the correct version.
    GOTO :EOF
) ELSE (
    ECHO Python version check passed.
)

ECHO.
ECHO --- Setting up Virtual Environment ('%VENV_NAME%') ---
IF NOT EXIST "%VENV_NAME%" (
    ECHO Creating virtual environment: %VENV_NAME%...
    REM Assuming 'python' command found above is the correct one.
    python -m venv "%VENV_NAME%"
    IF ERRORLEVEL 1 (
        ECHO Error: Failed to create virtual environment.
        GOTO :EOF
    )
    ECHO Virtual environment created.
) ELSE (
    ECHO Virtual environment '%VENV_NAME%' already exists.
)

ECHO.
ECHO --- Activating Virtual Environment and Installing Dependencies ---
REM Activate virtual environment
CALL "%VENV_NAME%\Scriptsctivate.bat"
IF ERRORLEVEL 1 (
    ECHO Error: Failed to activate virtual environment.
    ECHO Try activating manually: CALL %VENV_NAME%\Scriptsctivate.bat
    GOTO :EOF
)
ECHO Virtual environment activated.

REM Install dependencies from requirements.txt
IF EXIST "requirements.txt" (
    ECHO Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        ECHO Error: Failed to install dependencies from requirements.txt.
        REM Decide if script should exit or continue
    ) ELSE (
        ECHO Dependencies installed successfully.
    )
) ELSE (
    ECHO Warning: requirements.txt not found in the current directory. Skipping dependency installation.
)

ECHO.
ECHO --- Setup Complete ---
ECHO Python virtual environment '%VENV_NAME%' is set up and activated.
ECHO To deactivate, run: %VENV_NAME%\Scripts\deactivate.bat
ECHO To re-activate in a new terminal, run: CALL %VENV_NAME%\Scriptsctivate.bat

:EOF
ENDLOCAL
```
