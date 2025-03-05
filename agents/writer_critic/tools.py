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
    """⚠️ IMPORTANT: SAVE YOUR WORK ⚠️
    
    Save the current draft to a file.
    
    As the writer agent, you MUST use this tool to save your work after creating or revising content.
    This ensures your work is preserved and can be accessed by other agents.
    
    When to use this tool:
    - After completing an initial draft
    - When you've made significant revisions to a draft
    - Before suggesting further work by another agent
    - When you want to preserve your progress
    
    The saved draft will include metadata showing you (writer_agent) as the creator and will
    track the iteration number automatically.
    
    Args:
        draft: The draft content to save (MUST be a markdown STRING, not a dict!)
        draft_name: Optional name for the draft file (default: uses iteration counter)
        
    Returns:
        Confirmation message with the file path and ID
    """
    global iteration_count
    
    if not draft:
        return "No draft content to save."
    
    try:
        # Initialize file manager
        file_manager = FileManager()
        
        # Increment iteration count
        iteration_count += 1
        
        # Use provided name or generate one with iteration count
        title = draft_name if draft_name else f"Draft_{iteration_count}"
        
        # Save the file using the file manager
        file_id = file_manager.save_file(
            content=draft,
            file_type="draft",
            title=title,
            metadata={
                "word_count": get_word_count(draft),
                "source": "writer",
                "agent_name": "writer_agent",
                "iteration": iteration_count
            }
        )
        
        # Get the file metadata to return the path
        file_data = file_manager.get_file(file_id)
        filepath = file_data["metadata"]["filepath"]
        
        return f"Draft saved successfully to {filepath} (ID: {file_id})"
    
    except Exception as e:
        return f"Error saving draft: {str(e)}" 