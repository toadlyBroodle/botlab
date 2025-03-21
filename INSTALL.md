# Installing BotLab Dependencies

This directory contains a script to easily install the Python dependencies required by BotLab.

## Quick Installation

Run the following commands to install all dependencies:

```bash
# Make the script executable (if needed)
chmod +x install_dependencies.sh

# Run the installation script
./install_dependencies.sh
```

## Usage from Any Directory

You can run the installation script from any directory:

```bash
# Run directly from any path
/path/to/botlab/install_dependencies.sh
```

## What Does It Do?

The script:
1. Automatically finds and parses the dependencies in `agents/pyproject.toml`
2. Installs all dependencies using pip
3. Skips the Python version requirement

## Troubleshooting

If you encounter permission issues, try running with sudo:

```bash
sudo ./install_dependencies.sh
```

For virtual environment usage:

```bash
# Create and activate a virtual environment first
python -m venv .venv
source .venv/bin/activate

# Then run the script
./install_dependencies.sh
``` 