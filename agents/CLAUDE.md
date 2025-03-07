# CLAUDE.md - Agent Project Guidelines

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local imports last
- **Typing**: Use type hints for function parameters and return values
- **Docstrings**: Google style docstrings for all functions and classes
- **Error Handling**: Use try/except with specific exception types
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Agent Structure**: Create tools in tools.py, agent definitions in agents.py, run in main.py

## Project Organization
- Agents folder contains multiple agent systems (researcher, manager, writer-critic, editor, qaqc)
- Each agent system has its own tools.py, agents.py, and example.py
- Each agent has its own data/ directory for storing outputs
- Shared utilities in utils/ directory
- File management handled by utils/file_manager/

## File Management
- Each agent has its own data directory (e.g., researcher/data/)
- Files are saved with human-readable date prefixes (YYYY-MM-DD_HH-MM)
- The FileManager handles saving and loading files from the correct locations
- Use `load_file(agent_type)` to load the latest file from an agent
- Use `save_final_answer(agent, result, query_or_prompt, agent_type)` to save outputs
- PDF files are temporarily saved to researcher/data/papers/ and deleted after conversion to markdown

## Project Structure
### Main directory tree
```
.../agents/
    |__ .venv/
    |__ researcher/
    |   |__ data/
    |   |   |__ papers/
    |__ manager/
    |   |__ data/
    |__ editor/
    |   |__ data/
    |__ utils/
    |   |__ file_manager/
    |   |__ gemini/
    |__ writer_critic/
    |   |__ data/
    |__ qaqc/
        |__ data/
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
    - *file_manager/* contains the FileManager for centralized file handling
    - *gemini/* contains google gemini llm api utils
- *researcher/*, *manager/*, *writer_critic/*, *editor/*, and *qaqc/* are separate smolagent projects
    - smolagent projects have these common files:
        - `__init__.py`
        - `agents.py` : project agents
        - `example.py` : example usages
        - `tools.py` : tools used by the agents
        - `data/` : directory for storing agent outputs with human-readable date prefixes