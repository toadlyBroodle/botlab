import os
from datetime import datetime
import pytz
from smolagents import tool
from utils.agents.tools import get_timestamp
from utils.file_manager import FileManager
# Global variable to track iterations
iteration_count = 0

def get_word_count(text):
    """Count words in text"""
    return len(text.split())

@tool
def save_draft(draft: str, draft_name: str = None) -> str:
    """Save the current draft to a file.
    
    Args:
        draft: The draft content to save (MUST be a markdown STRING, not a dict!)
        draft_name: Optional name for the draft file (default: uses iteration counter)
        
    Returns:
        Confirmation message with the file path
    """
    global iteration_count
    
    if not draft:
        return "No draft content to save."
    
    # Initialize file manager
    file_manager = FileManager()
    
    # Use provided name or default to iteration number
    title = draft_name if draft_name else f"draft_{iteration_count}"
    
    # Save the file using the file manager
    file_id = file_manager.save_file(
        content=draft,
        file_type="draft",
        title=title,
        metadata={"word_count": get_word_count(draft), "iteration": iteration_count}
    )
    
    # Get the file metadata to return the path
    file_data = file_manager.get_file(file_id)
    file_path = file_data["metadata"]["filepath"]
    
    # Increment iteration count for next draft
    iteration_count += 1
    
    message = f"Draft saved to {file_path}"
    print(message)
    return message 