import os
import re
from datetime import datetime
from pathlib import Path
from smolagents import tool
from utils.agents.tools import get_timestamp
from utils.file_manager import FileManager

@tool
def save_draft(content: str, title: str = None, version: str = None) -> str:
    """Save draft content as a markdown file with timestamp, optional title, and version
    
    Args:
        content: The content to save
        title: Optional title for the file
        version: Optional version identifier
        
    Returns:
        Confirmation message with the file path
    """
    try:
        # Initialize file manager
        file_manager = FileManager()
        
        # Save the file using the file manager (as draft type)
        file_id = file_manager.save_file(
            content=content,
            file_type="draft",
            title=title,
            metadata={"word_count": len(content.split()), "source": "editor"},
            version=version
        )
        
        # Get the file metadata to return the path
        file_data = file_manager.get_file(file_id)
        filepath = file_data["metadata"]["filepath"]
        
        return f"Draft saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving draft: {str(e)}" 