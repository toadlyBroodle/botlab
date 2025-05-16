#!/bin/bash
# Script to install dependencies from pyproject.toml using Poetry and export requirements.txt
# This script assumes pyproject.toml is in the same directory as the script (e.g., botlab root).

set -e  # Exit on error

VERBOSE_POETRY_FLAG=""

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE_POETRY_FLAG="-v" # Pass -v to poetry; it can be stacked (e.g., -vv, -vvv for more verbosity)
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-v|--verbose]"
      echo "  -v, --verbose: Enable Poetry's verbose output"
      exit 1
      ;;
  esac
done

# Get the directory of this script, which is assumed to be the project root (e.g., botlab/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" # Ensure we are in the project root

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
    echo "Error: poetry command not found. Please install poetry first."
    echo "You can typically install it with: pip install poetry"
    echo "Or see the official documentation: https://python-poetry.org/docs/#installation"
    exit 1
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
# The output file "requirements.txt" will be created in the current directory ($SCRIPT_DIR)
# The $VERBOSE_POETRY_FLAG will add verbosity if enabled
if poetry export -f requirements.txt --output "requirements.txt" --without-hashes $VERBOSE_POETRY_FLAG; then
    echo "Successfully exported requirements.txt to $SCRIPT_DIR/requirements.txt"
else
    echo "Error: Poetry export failed. Check poetry output for details."
    exit 1
fi

echo "All operations (install and export) completed successfully!" 