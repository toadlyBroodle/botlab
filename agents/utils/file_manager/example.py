#!/usr/bin/env python3
"""
Example script demonstrating how to use the FileManager.

This script shows basic usage of the FileManager for saving, retrieving,
listing, and searching files across different agent types.
"""

import os
from pathlib import Path

from utils.file_manager import FileManager

def main():
    """Demonstrate FileManager functionality."""
    print("FileManager Example")
    print("==================")
    
    # Create a file manager with a project ID
    file_manager = FileManager(project_id="example_project")
    print(f"Created FileManager with project ID: {file_manager.project_id}")
    
    # Save files of different types
    print("\nSaving files...")
    
    # Save a draft from editor
    draft_content = """# Draft Document
    
This is an example draft document created by the FileManager.

## Features
- Easy file management
- Metadata tracking
- Project organization

*This content has been drafted for clarity.*
"""
    draft_id = file_manager.save_file(
        content=draft_content,
        file_type="draft",
        title="Example Draft",
        metadata={"tags": ["example", "draft"], "source": "editor"},
    )
    print(f"Saved draft with ID: {draft_id}")
    
    # Save a report
    report_content = """# Research Report
    
## Introduction

This is an example research report created by the FileManager.

## Findings

- Finding 1
- Finding 2
- Finding 3

## Conclusion

This demonstrates how to save and retrieve files across different agent types.
"""
    report_id = file_manager.save_file(
        content=report_content,
        file_type="report",
        title="Example Research Report",
        metadata={"tags": ["example", "research"]}
    )
    print(f"Saved report with ID: {report_id}")
    
    # Save another draft
    another_draft_content = """# Another Draft Document
    
This is an example draft document created by the FileManager.

It demonstrates how to save and retrieve files across different agent types.

*This content has been drafted for clarity.*
"""
    another_draft_id = file_manager.save_file(
        content=another_draft_content,
        file_type="draft",
        title="Another Example Draft",
        metadata={"tags": ["example", "draft"], "source": "editor"},
        version="v1"
    )
    print(f"Saved another draft with ID: {another_draft_id}")
    
    # Retrieve a file
    print("\nRetrieving files...")
    draft_data = file_manager.get_file(draft_id)
    print(f"Retrieved draft: {draft_data['metadata']['title']}")
    print(f"Draft path: {draft_data['metadata']['filepath']}")
    print(f"Draft word count: {len(draft_data['content'].split())}")
    
    # List files
    print("\nListing files...")
    all_files = file_manager.list_files()
    print(f"Total files: {len(all_files)}")
    
    # List files by type
    drafts = file_manager.list_files(file_type="draft")
    print(f"Draft files: {len(drafts)}")
    
    # List files by type and filter criteria
    writer_drafts = file_manager.list_files(file_type="draft", filter_criteria={"source": "writer"})
    print(f"Writer drafts: {len(writer_drafts)}")
    
    editor_drafts = file_manager.list_files(file_type="draft", filter_criteria={"source": "editor"})
    print(f"Editor drafts: {len(editor_drafts)}")
    
    reports = file_manager.list_files(file_type="report")
    print(f"Report files: {len(reports)}")
    
    # Search files
    print("\nSearching files...")
    search_results = file_manager.search_files("research")
    print(f"Files containing 'research': {len(search_results)}")
    for result in search_results:
        print(f"- {result['title']} ({result['file_type']})")
    
    print("\nFileManager example completed successfully!")

if __name__ == "__main__":
    main() 