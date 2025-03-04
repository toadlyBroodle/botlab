import os
import re
from datetime import datetime
from pathlib import Path
from smolagents import tool
from agents.utils.agents.tools import get_timestamp

# Set up paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
EDITS_DIR = BASE_DIR / "data" / "edits"

# Create directories if they don't exist
(BASE_DIR / "data").mkdir(exist_ok=True)
EDITS_DIR.mkdir(exist_ok=True)

@tool
def save_edit(content: str, title: str = None, version: str = None) -> str:
    """Save edited content as a markdown file with timestamp, optional title, and version
    
    Args:
        content: The content to save
        title: Optional title to include in the filename
        version: Optional version string (e.g., "v1", "v2")
        
    Returns:
        A message indicating the file was saved successfully
    """
    try:
        # Generate timestamp
        timestamp = get_timestamp()
        
        # Clean title for filename
        if title:
            # Remove special characters and replace spaces with underscores
            clean_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            filename = f"{timestamp}_{clean_title}"
        else:
            filename = timestamp
            
        # Add version if provided
        if version:
            filename = f"{filename}_{version}"
            
        # Complete filename with extension
        filename = f"{filename}.md"
        
        # Full path to save
        filepath = EDITS_DIR / filename
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Edit saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving edit: {str(e)}" 