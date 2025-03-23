#!/bin/bash

# Setup script for creating a root .venv directory for the entire project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up virtual environment in root directory..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
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

echo "Setup complete! To activate the environment, run: source .venv/bin/activate" 