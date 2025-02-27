# Agent Systems

This directory contains agent systems built with smolagents:

- **researcher**: A web search and information gathering agent
- **manager**: A flexible coordination agent that can manage multiple specialized agents
- **writer-critic**: A creative writing system with feedback and iteration

## Running the Examples

### Option 1: Using the run_examples.py script

From the `agents` directory:

```bash
# Run the researcher example
poetry run python run_examples.py researcher "your query"

# Run the basic manager example (with researcher agent)
poetry run python run_examples.py manager "your query"

# Run the advanced manager example (with custom agent configuration)
poetry run python run_examples.py manager-advanced "your query"

# Run the manager with custom agent selection
poetry run python run_examples.py manager-custom "your query" --agents researcher writer

# Run the writer-critic example
poetry run python run_examples.py writer

# Run all examples
poetry run python run_examples.py all

# Enable telemetry for any example
poetry run python run_examples.py researcher "your query" --telemetry
```

### Option 2: Running individual examples as modules

From the `agents` directory:

```bash
# Run the researcher example
poetry run python -m researcher.example "your query"

# Run the manager example
poetry run python -m manager.example "your query"

# Run the manager with advanced setup
poetry run python -m manager.example "your query" --advanced

# Run the manager with specific agent types
poetry run python -m manager.example "your query" --no-researcher --writer --agents researcher writer

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
