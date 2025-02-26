# Agent Systems

This directory contains agent systems built with smolagents:

- **researcher**: A web search and information gathering agent system
- **writer-critic**: A creative writing system with feedback and iteration

## Running the Examples

### Option 1: Using the run_examples.py script

From the `agents` directory:

```bash
# Run the researcher example
poetry run python run_examples.py researcher

# Run the writer-critic example
poetry run python run_examples.py writer

# Run both examples
poetry run python run_examples.py all
```

### Option 2: Running individual examples as modules

From the `agents` directory:

```bash
# Run the researcher example
poetry run python -m researcher.example

# Run the writer-critic example
poetry run python -m writer_critic.example
```

## Dependencies

Make sure you have all required dependencies installed:

```bash
pip install smolagents dotenv
```

Or if using Poetry:

```bash
poetry install
```
