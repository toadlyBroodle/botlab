# CLAUDE.md - Agent Project Guidelines

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local imports last
- **Typing**: Use type hints for function parameters and return values
- **Docstrings**: Google style docstrings for all functions and classes
- **Error Handling**: Use try/except with specific exception types
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Agent Structure**: Create tools in tools.py, agent definitions in agents.py, run in main.py

## Project Organization
- Agents folder contains multiple agent systems (researcher, manager, writer-critic)
- Each agent system has its own tools.py, agents.py, and main.py
- Shared utilities in utils/ directory
- Examples run via run_examples.py

## Project Structure (.prompt)
### Main directory tree
```
.../agents/
    |__ .venv/
    |__ researcher/
    |__ manager/
    |__ editor/
    |__ utils/
    |   |__ gemini/
    |__ writer_critic/
```

### Virtual environment
- *.venv/* is the poetry managed .venv
- All python scripts MUST be prefixed by `poetry run python `
    - e.g.  `poetry run python -m researcher.example`
            `poetry run python researcher/example.py`

### smolagent projects
- Working directory is *agents/*
    - All python scripts MUST be run from *agents/* working directory:
    ```
    cd agents
    poetry run python -m researcher.example
    ```
- *utils/* contains shared utils for the project
    - *gemini/* contains google gemini llm api utils
- *researcher/*, *manager/*, *writer_critic/*, and *editor/* are separate smolagent projects
    - smolagent projects have these common files:
        - `__init__.py`
        - `agents.py` : project agents
        - `example.py` : example usages
        - `main.py` : main setup and entry point
        - `tools.py` : tools used by the agents