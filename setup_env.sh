#!/bin/bash

# Script to initialize a Python virtual environment and install dependencies.

VENV_NAME="cr"
PYTHON_MAJOR_REQUIRED=3
PYTHON_MINOR_REQUIRED=12
PYTHON_COMMAND="python${PYTHON_MAJOR_REQUIRED}.${PYTHON_MINOR_REQUIRED}" # e.g., python3.12
FALLBACK_PYTHON_COMMAND="python3" # If specific version command isn't found

# Function to check Python version
check_python_version() {
    local python_cmd_to_test="$1"
    if ! command -v "$python_cmd_to_test" &> /dev/null; then
        # echo "Python command '$python_cmd_to_test' not found." # Suppress for cleaner flow
        return 1
    fi

    # Get version string (e.g., "Python 3.12.1")
    local version_str
    version_str=$("$python_cmd_to_test" --version 2>&1) # Get version, redirect stderr to stdout

    if [[ "$version_str" =~ Python\ ([0-9]+)\.([0-9]+) ]]; then
        local major=${BASH_REMATCH[1]}
        local minor=${BASH_REMATCH[2]}
        if [ "$major" -gt "$PYTHON_MAJOR_REQUIRED" ] ||            ( [ "$major" -eq "$PYTHON_MAJOR_REQUIRED" ] && [ "$minor" -ge "$PYTHON_MINOR_REQUIRED" ] ); then
            echo "Found suitable Python: $python_cmd_to_test ($version_str)"
            PYTHON_EXECUTABLE="$python_cmd_to_test" # Set the global PYTHON_EXECUTABLE
            return 0 # Success
        else
            # echo "Python version $major.$minor from '$python_cmd_to_test' is too old. Required: $PYTHON_MAJOR_REQUIRED.$PYTHON_MINOR_REQUIRED+"
            return 1
        fi
    else
        # echo "Could not parse Python version string from '$python_cmd_to_test': $version_str"
        return 1
    fi
}

echo "--- Checking Python Version (Required: ${PYTHON_MAJOR_REQUIRED}.${PYTHON_MINOR_REQUIRED}+) ---"
PYTHON_EXECUTABLE="" # Global variable to store the found python command

# Try specific version command first (e.g., python3.12)
if check_python_version "$PYTHON_COMMAND"; then
    : # PYTHON_EXECUTABLE is set by check_python_version
elif check_python_version "$FALLBACK_PYTHON_COMMAND"; then
    : # PYTHON_EXECUTABLE is set by check_python_version
else
    echo "Error: Python ${PYTHON_MAJOR_REQUIRED}.${PYTHON_MINOR_REQUIRED}+ not found."
    echo "Please install Python ${PYTHON_MAJOR_REQUIRED}.${PYTHON_MINOR_REQUIRED} or newer and ensure it's in your PATH."
    echo "Tried commands: '$PYTHON_COMMAND', '$FALLBACK_PYTHON_COMMAND'."
    exit 1
fi

echo ""
echo "--- Setting up Virtual Environment ('$VENV_NAME') ---"
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME using $PYTHON_EXECUTABLE..."
    "$PYTHON_EXECUTABLE" -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created."
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

echo ""
echo "--- Activating Virtual Environment and Installing Dependencies ---"
# Activate virtual environment
source "$VENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    echo "Try activating manually: source $VENV_NAME/bin/activate"
    exit 1
fi
echo "Virtual environment activated."

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies from requirements.txt."
        # Consider exiting here or allowing script to continue
    else
        echo "Dependencies installed successfully."
    fi
else
    echo "Warning: requirements.txt not found in the current directory. Skipping dependency installation."
fi

echo ""
echo "--- Setup Complete ---"
echo "Python virtual environment '$VENV_NAME' is set up and activated."
echo "To deactivate, run: deactivate"
echo "To re-activate in a new terminal, run: source $VENV_NAME/bin/activate"
