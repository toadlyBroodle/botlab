# Installing BotLab Dependencies

This directory contains a script to easily set up a virtual environment and install all Python dependencies required by BotLab.

## Quick Installation

Run the following commands to set up the environment and install all dependencies:

```bash
# Make the script executable (if needed)
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

## What Does It Do?

The script:
1. Creates a `.venv` directory at the root of the project
2. Installs all dependencies from `requirements.txt`
3. Installs local packages (`agents` and `swarms`) in development mode

## Usage

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

If you encounter permission issues, make sure the script is executable:

```bash
chmod +x setup_env.sh
```

For manual installation:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install local packages
pip install -e ./agents
pip install -e ./swarms
``` 

If you prefer to use Poetry you may use the `poetry.toml` file to install the dependencies.
