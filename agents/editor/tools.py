import os
import re
from datetime import datetime
from pathlib import Path
from smolagents import tool
from utils.agents.tools import get_timestamp
from utils.file_manager import FileManager

@tool
def save_draft(content: str, title: str = None, version: str = None) -> str:
    """⚠️ IMPORTANT: SAVE YOUR WORK ⚠️
    
    Save draft content as a markdown file with timestamp, optional title, and version.
    
    As the editor agent, you MUST use this tool to save your work after making significant edits.
    This ensures your work is preserved and can be accessed by other agents.
    
    When to use this tool:
    - After completing a round of edits
    - When you've made significant improvements to a draft
    - Before suggesting further work by another agent
    - When you want to preserve a version before making major changes
    
    The saved draft will include metadata showing you (editor_agent) as the creator.
    
    Args:
        content: The content to save (MUST be a markdown STRING)
        title: Optional title for the file
        version: Optional version identifier
        
    Returns:
        Confirmation message with the file path and ID
    """
    try:
        # Initialize file manager
        file_manager = FileManager()
        
        # Save the file using the file manager (as draft type)
        file_id = file_manager.save_file(
            content=content,
            file_type="draft",
            title=title,
            metadata={
                "word_count": len(content.split()), 
                "source": "editor",
                "agent_name": "editor_agent",
                "version": version
            },
            version=version
        )
        
        # Get the file metadata to return the path
        file_data = file_manager.get_file(file_id)
        filepath = file_data["metadata"]["filepath"]
        
        return f"Draft saved successfully to {filepath} (ID: {file_id})"
    except Exception as e:
        return f"Error saving draft: {str(e)}" 