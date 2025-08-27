#!/bin/bash

# Setup script for creating a root .venv directory for the entire project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up virtual environment in root directory..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3.13 -m venv .venv
    echo "Created new virtual environment in .venv"
else
    echo "Virtual environment already exists in .venv"
fi

# Activate virtual environment and install dependencies
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Temporarily rename pyproject.toml files to avoid conflicts with setup.py
echo "Temporarily moving pyproject.toml files..."
if [ -f "$SCRIPT_DIR/agents/pyproject.toml" ]; then
    mv "$SCRIPT_DIR/agents/pyproject.toml" "$SCRIPT_DIR/agents/pyproject.toml.bak"
fi
if [ -f "$SCRIPT_DIR/swarms/pyproject.toml" ]; then
    mv "$SCRIPT_DIR/swarms/pyproject.toml" "$SCRIPT_DIR/swarms/pyproject.toml.bak"
fi

# Install the local packages in development mode
echo "Installing local packages in development mode..."
cd "$SCRIPT_DIR/agents"
pip install -e .
# Uncomment this to install swarms
#cd "$SCRIPT_DIR/swarms"
#pip install -e .
cd "$SCRIPT_DIR"

# Restore pyproject.toml files for reference
echo "Restoring pyproject.toml files..."
if [ -f "$SCRIPT_DIR/agents/pyproject.toml.bak" ]; then
    mv "$SCRIPT_DIR/agents/pyproject.toml.bak" "$SCRIPT_DIR/agents/pyproject.toml"
fi
if [ -f "$SCRIPT_DIR/swarms/pyproject.toml.bak" ]; then
    mv "$SCRIPT_DIR/swarms/pyproject.toml.bak" "$SCRIPT_DIR/swarms/pyproject.toml"
fi

# Ask user if they want to fix smolagents code blob parsing
echo ""
read -p "Do you want to fix smolagents code blob parsing to handle python code blocks to fix the persistent code block parsing error (highly recommended)? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Fixing smolagents code blob parsing..."
    SMOLAGENTS_UTILS=".venv/lib/python3.12/site-packages/smolagents/utils.py"
    if [ -f "$SMOLAGENTS_UTILS" ]; then
        # Replace the regex pattern for better code block parsing
        sed -i 's/pattern = r"<code>(.*?)<\/code>"/pattern = r"Code:\\n*```(?:py|python)?\\n(.*?)\\n```<end_code>"/' "$SMOLAGENTS_UTILS"
        echo "Successfully fixed smolagents code blob parsing pattern"
    else
        echo "Warning: smolagents utils.py not found at expected location"
    fi
else
    echo "Skipping smolagents code blob parsing fix"
fi

echo "Setup complete! To activate the environment, run: source .venv/bin/activate" 