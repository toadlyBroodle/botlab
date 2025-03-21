#!/bin/bash
# Script to install dependencies from botlab/agents/pyproject.toml by parsing the TOML file

set -e  # Exit on error

# Parse command line options
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-v|--verbose]"
      echo "  -v, --verbose: Show detailed output"
      exit 1
      ;;
  esac
done

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Installing botlab dependencies from $SCRIPT_DIR..."

# Check if the pyproject.toml file exists
PYPROJECT_PATH="$SCRIPT_DIR/agents/pyproject.toml"
if [ ! -f "$PYPROJECT_PATH" ]; then
    echo "Error: agents/pyproject.toml not found at $PYPROJECT_PATH!"
    exit 1
fi

echo "Parsing dependencies from pyproject.toml..."

# Create a temporary file to store the dependency section
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

# Extract the dependencies section from pyproject.toml
# First, determine start line of dependencies section
DEP_START_LINE=$(grep -n "\[tool.poetry.dependencies\]" "$PYPROJECT_PATH" | cut -d: -f1)

if [ -z "$DEP_START_LINE" ]; then
    echo "Error: Could not find dependencies section in pyproject.toml"
    exit 1
fi

# Find the next section after dependencies to determine end line
NEXT_SECTION_LINE=$(tail -n +"$DEP_START_LINE" "$PYPROJECT_PATH" | grep -n "^\[" | grep -v "\[tool.poetry.dependencies\]" | head -1 | cut -d: -f1)

if [ -z "$NEXT_SECTION_LINE" ]; then
    # If there's no next section, use the end of file
    DEP_END_LINE=$(wc -l "$PYPROJECT_PATH" | awk '{print $1}')
else
    # Calculate actual line number by adding DEP_START_LINE and NEXT_SECTION_LINE, then subtract 1
    DEP_END_LINE=$((DEP_START_LINE + NEXT_SECTION_LINE - 1))
fi

# Extract the dependency lines to the temp file
sed -n "${DEP_START_LINE},${DEP_END_LINE}p" "$PYPROJECT_PATH" > "$TEMP_FILE"

# Display the extracted section in verbose mode
if [ "$VERBOSE" = true ]; then
    echo "Extracted dependency section (lines $DEP_START_LINE-$DEP_END_LINE):"
    cat "$TEMP_FILE"
    echo "------------------------------------------------------------"
fi

# Process the dependencies: find lines with '=', exclude python version, and extract package names
DEPENDENCIES=$(cat "$TEMP_FILE" | 
               grep "=" | 
               grep -v "^\[" |
               grep -v "python =" | 
               awk -F '=' '{print $1}' | 
               sed 's/^[[:space:]]*//g' | 
               sed 's/[[:space:]]*$//g')

# Check if we found any dependencies
if [ -z "$DEPENDENCIES" ]; then
    echo "Error: No dependencies found in pyproject.toml!"
    echo "Here's the content of the file:"
    cat "$PYPROJECT_PATH"
    exit 1
fi

# Count the dependencies for reporting
DEP_COUNT=$(echo "$DEPENDENCIES" | wc -l)
echo "Found $DEP_COUNT Python packages to install..."

if [ "$VERBOSE" = true ]; then
    echo "Packages to install:"
    echo "$DEPENDENCIES"
    echo "------------------------------------------------------------"
fi

# Install each dependency one by one to avoid problems with shell expansion
echo "Installing dependencies from pyproject.toml..."
FAILED_PACKAGES=""
while read -r package; do
    echo "Installing $package..."
    if ! pip install "$package"; then
        echo "Warning: Failed to install $package"
        FAILED_PACKAGES="$FAILED_PACKAGES $package"
    fi
done <<< "$DEPENDENCIES"

# Report on any failures
if [ -n "$FAILED_PACKAGES" ]; then
    echo "Warning: Some packages failed to install:$FAILED_PACKAGES"
    echo "You may need to install them manually."
    exit 1
else
    echo "All dependencies installed successfully!"
fi 