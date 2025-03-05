# File Manager

A centralized file management system for sharing files across all agents and multiple runs.

## Overview

The File Manager provides a unified way to save, retrieve, list, and search files across different agent types. It ensures consistent file naming, organization, and metadata tracking.

## Features

- **Centralized Storage**: All files are stored in a shared directory structure
- **Consistent Naming**: Files are named with timestamp, project ID, and title
- **Metadata Tracking**: Each file has associated metadata for easy retrieval and filtering
- **Project Organization**: Files can be grouped by project ID
- **File Types**: Support for different file types (drafts, reports, resources, etc.)
- **Search Capability**: Search for files by content or metadata

## Directory Structure

The FileManager uses the following directory structure:

```
/shared_data/
  /drafts/        # Writer drafts and editor drafts
  /reports/       # Researcher reports and final documents
  /resources/     # Downloaded resources and cached data
  /archive/       # Archived files
  /research/
    /papers/      # Research papers (PDFs and converted markdown)
  metadata_index.json  # Metadata for all files
```

## Usage

### Basic Usage

```python
from utils.file_manager import FileManager

# Initialize with optional project ID
file_manager = FileManager(project_id="my_project")

# Save a file
file_id = file_manager.save_file(
    content="# My Document\n\nThis is the content.",
    file_type="draft",
    title="My Document Title",
    metadata={"tags": ["example", "documentation"]}
)

# Retrieve a file
file_data = file_manager.get_file(file_id)
content = file_data["content"]
metadata = file_data["metadata"]

# List files
all_files = file_manager.list_files()
draft_files = file_manager.list_files(file_type="draft")

# Search files
results = file_manager.search_files("example")
```

### File Types

The FileManager supports the following file types:

- `draft`: Writer drafts and editor drafts
- `report`: Researcher reports and final documents
- `resource`: Downloaded resources and cached data
- `archive`: Archived files
- `paper`: Research papers (PDFs and converted markdown)

### Metadata

Each file has the following metadata:

- `file_id`: Unique identifier
- `filename`: The filename on disk
- `filepath`: Full path to the file
- `file_type`: Type of file
- `title`: Optional title
- `project_id`: Project identifier
- `created_at`: Creation timestamp
- `version`: Optional version identifier
- `extension`: File extension
- Custom metadata: Any additional metadata provided when saving

## Example

See `example.py` for a complete example of using the File Manager. 