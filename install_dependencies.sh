#!/bin/bash
# Script to install dependencies from pyproject.toml using Poetry and export requirements.txt
# This script assumes pyproject.toml is in the same directory as the script (e.g., botlab root).

set -e  # Exit on error

VERBOSE_POETRY_FLAG=""

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose|-vv|-vvv)
      VERBOSE_POETRY_FLAG=$1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-v|--verbose|-vv|-vvv]"
      echo "  -v, --verbose: Enable Poetry's verbose output"
      echo "  -vv, -vvv: Enable more verbose output"
      exit 1
      ;;
  esac
done

# Get the directory of this script, which is assumed to be the project root (e.g., botlab/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" # Ensure we are in the project root

# Check if python3.13 installed and configure Poetry to use it
if ! command -v python3.13 &> /dev/null; then
    echo "Error: python3.13 could not be found. Please install it first."
    exit 1
else
    echo "Python 3.13 found. Attempting to configure Poetry to use it..."
    if poetry env use python3.13 $VERBOSE_POETRY_FLAG; then
        echo "Poetry successfully configured to use python3.13."
    else
        echo "Error: 'poetry env use python3.13' failed."
        echo "Please check Poetry's output above for details. Make sure python3.13 is a functional installation that Poetry can use."
        exit 1
    fi
fi

echo "Running in directory: $SCRIPT_DIR"
echo "Looking for pyproject.toml in this directory..."

# Check if pyproject.toml file exists in the current directory (project root)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in $SCRIPT_DIR!"
    echo "Please ensure the file has been moved to the root of the 'botlab' directory."
    exit 1
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Do you want to install it? (y/n)"
    read -r install_poetry
    if [[ "$install_poetry" =~ ^[Yy]$ ]]; then
        echo "Installing Poetry using the official installer..."
        curl -sSL https://install.python-poetry.org | python3 -
        
        # Check if PATH is already in .bashrc
        if grep -q "PATH=\"\$HOME/.local/bin:\$PATH\"" ~/.bashrc; then
            echo "Poetry PATH already in .bashrc"
        else
            echo "Adding Poetry to PATH in .bashrc"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            export PATH="$HOME/.local/bin:$PATH"
            source ~/.bashrc
        fi
    else
        echo "Poetry installation skipped. Please install Poetry manually and run this script again."
        exit 1
    fi
fi

echo "Poetry found. Proceeding with operations using pyproject.toml in $SCRIPT_DIR."

# Install dependencies using Poetry
echo "Installing dependencies from pyproject.toml using 'poetry install'..."
# The $VERBOSE_POETRY_FLAG will add verbosity to the poetry command if -v was passed to the script
if poetry install $VERBOSE_POETRY_FLAG; then
    echo "Poetry install completed successfully."
else
    echo "Error: Poetry install failed. Check poetry output for details."
    exit 1
fi

# Export dependencies to requirements.txt
echo "Exporting dependencies to requirements.txt..."
poetry self add poetry-plugin-export
# The output file "requirements.txt" will be created in the current directory ($SCRIPT_DIR)
# The $VERBOSE_POETRY_FLAG will add verbosity if enabled
if poetry export -f requirements.txt --output "requirements.txt" --without-hashes $VERBOSE_POETRY_FLAG; then
    echo "Successfully exported requirements.txt to $SCRIPT_DIR/requirements.txt"
else
    echo "Error: Poetry export failed. Check poetry output for details."
    exit 1
fi

echo "All operations (install and export) completed successfully!" 