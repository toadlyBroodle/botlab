# File Manager

A centralized file management system for agent outputs with agent-specific data directories.

## Overview

The File Manager provides a unified way to save, retrieve, list, and search files across different agent types. It ensures consistent file naming and organization, and operates directly on the filesystem without a metadata index.

## Features

- **Agent-Specific Storage**: Each agent has its own data directory (e.g., researcher/data/)
- **Human-Readable Naming**: Files are named with human-readable date prefixes (YYYY-MM-DD_HH-MM)
- **File Types**: Support for different file types (reports, papers, etc.)
- **Search Capability**: Search for files by content or filename
- **CLI Commands**: Command-line interface for managing files

## Usage

### Basic Usage

```python
from agents.utils.file_manager import FileManager
from agents.utils.agents.tools import save_final_answer, load_file

# Initialize the FileManager
file_manager = FileManager()

# Save a file with metadata
file_id = file_manager.save_file(
    content="This is the content of my file.",
    metadata={
        "agent_name": "researcher_agent",
        "query": "Tell me about AI agents",
        "content_type": "research_notes"
    }
)

# List files with filtering
files = file_manager.list_files(
    filter_criteria={"agent_name": "researcher_agent"}
)

# Get file by ID
file_data = file_manager.get_file(file_id)
```

### Advanced Usage

```python
from agents.utils.file_manager import FileManager

# Initialize the FileManager
file_manager = FileManager()

# Save a file
file_path = file_manager.save_file(
    content="# My Document\n\nThis is the content.",
    file_type="report",
    title="My Document Title",
    agent_name="researcher_agent"
)

# Retrieve a file
file_data = file_manager.get_file(file_path)
content = file_data["content"]

# List files
all_files = file_manager.list_files()
agent_files = file_manager.list_files(filter_criteria={"agent_name": "researcher_agent"})

# Search files
results = file_manager.search_files("example", agent_name="researcher_agent")

# Delete files
file_manager.delete_file(file_path)
file_manager.delete_agent_files("researcher")
file_manager.delete_all_files()
```

### Command-Line Interface

The File Manager includes a command-line interface for managing files:

```bash
# List all files
poetry run python -m utils.file_manager.file_manager list

# List files for a specific agent
poetry run python -m utils.file_manager.file_manager list --agent researcher

# Delete a specific file
poetry run python -m utils.file_manager.file_manager delete --file-id <file_path>

# Delete all files for a specific agent
poetry run python -m utils.file_manager.file_manager delete --agent researcher

# Delete all files
poetry run python -m utils.file_manager.file_manager delete --all
```

### Special Case: PDF Handling

For the researcher agent, PDF files are:
- Temporarily saved to researcher/data/papers/
- Converted to markdown with a filename format: YYYY-MM-DD_HH-MM_paper_id.md
- Original PDFs are deleted after conversion
- Markdown files can be read using the `read_paper_markdown` function
