import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Import the timestamp function we already have
from utils.agents.tools import get_timestamp

# Base directory for shared data
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SHARED_DATA_DIR = BASE_DIR.parent / "shared_data"

# Ensure directories exist
DRAFTS_DIR = SHARED_DATA_DIR / "drafts"
REPORTS_DIR = SHARED_DATA_DIR / "reports"
RESOURCES_DIR = SHARED_DATA_DIR / "resources"
ARCHIVE_DIR = SHARED_DATA_DIR / "archive"
RESEARCH_PAPERS_DIR = SHARED_DATA_DIR / "research" / "papers"

# Create directories if they don't exist
for directory in [DRAFTS_DIR, REPORTS_DIR, RESOURCES_DIR, ARCHIVE_DIR]:
    directory.mkdir(exist_ok=True)

# Create research/papers directory with parents=True to ensure parent directories are created
RESEARCH_PAPERS_DIR.mkdir(exist_ok=True, parents=True)

# Path to the metadata index file
METADATA_INDEX = SHARED_DATA_DIR / "metadata_index.json"

class FileManager:
    """Centralized file manager for all agents to share files across runs."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize the file manager.
        
        Args:
            project_id: Optional project identifier to group related files
        """
        self.project_id = project_id or str(uuid.uuid4())[:8]
        self._load_metadata_index()
    
    def _load_metadata_index(self) -> None:
        """Load the metadata index from disk or create if it doesn't exist."""
        if METADATA_INDEX.exists():
            try:
                with open(METADATA_INDEX, 'r', encoding='utf-8') as f:
                    self.metadata_index = json.load(f)
            except json.JSONDecodeError:
                # If the file is corrupted, create a new index
                self.metadata_index = {"files": {}}
        else:
            self.metadata_index = {"files": {}}
    
    def _save_metadata_index(self) -> None:
        """Save the metadata index to disk."""
        with open(METADATA_INDEX, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_index, f, indent=2)
    
    def _get_directory_for_file_type(self, file_type: str) -> Path:
        """Get the appropriate directory for a given file type.
        
        Args:
            file_type: Type of file (draft, report, resource, archive, paper)
            
        Returns:
            Path to the directory
        """
        file_type = file_type.lower()
        if file_type == "draft":
            return DRAFTS_DIR
        elif file_type == "report":
            return REPORTS_DIR
        elif file_type == "resource":
            return RESOURCES_DIR
        elif file_type == "archive":
            return ARCHIVE_DIR
        elif file_type == "paper":
            return RESEARCH_PAPERS_DIR
        else:
            # Default to resources for unknown types
            return RESOURCES_DIR
    
    def save_file(self, 
                 content: str, 
                 file_type: str, 
                 title: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 extension: str = ".md",
                 version: Optional[str] = None) -> str:
        """Save a file to the appropriate directory with metadata.
        
        Args:
            content: The content to save
            file_type: Type of file (draft, report, resource, archive)
            title: Optional title for the file
            metadata: Optional metadata to store with the file
            extension: File extension (default: .md)
            version: Optional version identifier
            
        Returns:
            The file_id of the saved file
        """
        # Generate a unique file ID
        file_id = str(uuid.uuid4())
        
        # Get timestamp
        timestamp = get_timestamp()
        
        # Clean title for filename
        if title:
            # Remove special characters and replace spaces with underscores
            safe_title = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in title)
            safe_title = safe_title.replace(' ', '_')
            filename = f"{timestamp}_{self.project_id}_{safe_title}"
        else:
            filename = f"{timestamp}_{self.project_id}_{file_type}"
        
        # Add version if provided
        if version:
            filename = f"{filename}_v{version}"
        
        # Complete filename with extension
        if not extension.startswith('.'):
            extension = f".{extension}"
        filename = f"{filename}{extension}"
        
        # Get the appropriate directory
        directory = self._get_directory_for_file_type(file_type)
        
        # Full path to save
        filepath = directory / filename
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Prepare metadata
        file_metadata = {
            "file_id": file_id,
            "filename": filename,
            "filepath": str(filepath),
            "file_type": file_type,
            "title": title,
            "project_id": self.project_id,
            "created_at": datetime.now().isoformat(),
            "version": version,
            "extension": extension,
            **({} if metadata is None else metadata)
        }
        
        # Update metadata index
        self.metadata_index["files"][file_id] = file_metadata
        self._save_metadata_index()
        
        return file_id
    
    def get_file(self, file_identifier: str) -> Dict[str, Any]:
        """Get a file by its ID or filename.
        
        Args:
            file_identifier: The file ID or filename
            
        Returns:
            Dict containing file content and metadata
        """
        # Check if it's a file ID
        if file_identifier in self.metadata_index["files"]:
            file_metadata = self.metadata_index["files"][file_identifier]
            filepath = file_metadata["filepath"]
        else:
            # Search by filename
            found = False
            for file_id, metadata in self.metadata_index["files"].items():
                if metadata["filename"] == file_identifier:
                    file_metadata = metadata
                    filepath = metadata["filepath"]
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(f"File not found: {file_identifier}")
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "content": content,
            "metadata": file_metadata
        }
    
    def list_files(self, 
                  file_type: Optional[str] = None, 
                  project_id: Optional[str] = None,
                  filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List files with optional filtering.
        
        Args:
            file_type: Optional filter by file type
            project_id: Optional filter by project ID
            filter_criteria: Optional additional filter criteria
            
        Returns:
            List of file metadata matching the criteria
        """
        filter_criteria = filter_criteria or {}
        project_id = project_id or self.project_id
        
        results = []
        for file_id, metadata in self.metadata_index["files"].items():
            # Apply filters
            if file_type and metadata["file_type"] != file_type:
                continue
                
            if project_id and metadata["project_id"] != project_id:
                continue
            
            # Apply additional filters
            skip = False
            for key, value in filter_criteria.items():
                if key not in metadata or metadata[key] != value:
                    skip = True
                    break
            
            if not skip:
                results.append(metadata)
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x["created_at"], reverse=True)
        
        return results
    
    def delete_file(self, file_identifier: str) -> bool:
        """Delete a file by its ID or filename.
        
        Args:
            file_identifier: The file ID or filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get file metadata
            file_data = self.get_file(file_identifier)
            file_metadata = file_data["metadata"]
            
            # Delete the file
            os.remove(file_metadata["filepath"])
            
            # Remove from metadata index
            del self.metadata_index["files"][file_metadata["file_id"]]
            self._save_metadata_index()
            
            return True
        except (FileNotFoundError, KeyError):
            return False
    
    def search_files(self, query: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for files containing the query in title or content.
        
        Args:
            query: Search query
            file_type: Optional filter by file type
            
        Returns:
            List of file metadata matching the query
        """
        query = query.lower()
        results = []
        
        # First get all files that match the file_type filter
        files = self.list_files(file_type=file_type)
        
        for file_metadata in files:
            # Check if query is in title
            title = file_metadata.get("title", "").lower()
            if query in title:
                results.append(file_metadata)
                continue
            
            # Check if query is in content
            try:
                with open(file_metadata["filepath"], 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if query in content:
                        results.append(file_metadata)
            except Exception:
                # Skip files that can't be read
                continue
        
        return results 