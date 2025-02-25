import os
from datetime import datetime
import pytz
from smolagents import tool

# Ensure drafts directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.join(BASE_DIR, "drafts")
os.makedirs(DRAFT_DIR, exist_ok=True)

# Global variable to track iterations
iteration_count = 0

def get_timestamp():
    """Get current timestamp in Pacific timezone"""
    return datetime.now(pytz.timezone('US/Pacific')).strftime("%Y%m%d_%H%M")

def get_word_count(text):
    """Count words in text"""
    return len(text.split())

@tool
def save_draft(draft: str, draft_name: str = None) -> str:
    """Save the current draft to a file.
    
    Args:
        draft: The text content to save
        draft_name: Optional name for the draft file (default: uses iteration counter)
        
    Returns:
        Confirmation message with the file path
    """
    global iteration_count
    
    if not draft:
        return "No draft content to save."
    
    if not os.path.exists(DRAFT_DIR):
        os.makedirs(DRAFT_DIR, exist_ok=True)

    # Use provided name or default to iteration number
    if draft_name:
        file_name = f"{draft_name}.md"
    else:
        file_name = f"draft_{iteration_count}.md"
    
    file_path = os.path.join(DRAFT_DIR, file_name)
    with open(file_path, "w") as f:
        f.write(draft)
    
    # Increment iteration count for next draft
    iteration_count += 1
    
    message = f"Draft saved to {file_path}"
    print(message)
    return message 