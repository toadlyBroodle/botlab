import os
import json
import uuid
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

# Import the timestamp function
from ..agents.tools import get_timestamp

# Base directory for the project - points to agents/
BASE_DIR = Path(__file__).parent.parent.parent

# File manager directory
FILE_MANAGER_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Agent-specific data directories
AGENT_DIRS = {
    "researcher_agent": BASE_DIR / "researcher" / "data",
    "manager_agent": BASE_DIR / "manager" / "data",
    "editor_agent": BASE_DIR / "editor" / "data",
    "writer_critic_agent": BASE_DIR / "writer_critic" / "data",
    "qaqc_agent": BASE_DIR / "qaqc" / "data",
}

# Create researcher papers directory
RESEARCHER_PAPERS_DIR = AGENT_DIRS["researcher_agent"] / "papers"
RESEARCHER_PAPERS_DIR.mkdir(exist_ok=True, parents=True)

# Create agent-specific data directories
for agent_dir in AGENT_DIRS.values():
    agent_dir.mkdir(exist_ok=True, parents=True)

# Configure logging
logger = logging.getLogger("file_manager")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class FileManager:
    """File manager for saving and loading agent outputs."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize the file manager.
        
        Args:
            project_id: Optional project identifier to group related files
        """
        self.project_id = project_id or str(uuid.uuid4())[:8]
    
    def _get_directory_for_agent(self, agent_name: Optional[str] = None, file_type: Optional[str] = None) -> Path:
        """Get the appropriate directory for a given agent.
        
        Args:
            agent_name: Name of the agent (e.g., "researcher_agent")
            file_type: Type of file (e.g., "paper")
            
        Returns:
            Path to the directory
        """
        # Special case for researcher papers
        if file_type == "paper":
            return RESEARCHER_PAPERS_DIR
        
        # Otherwise use the agent's data directory
        if agent_name and agent_name in AGENT_DIRS:
            return AGENT_DIRS[agent_name]
        
        # Default to researcher if no agent specified
        return AGENT_DIRS["researcher_agent"]
    
    def save_file(self, 
                 content: str, 
                 file_type: str = "report", 
                 title: Optional[str] = None, 
                 agent_name: Optional[str] = None,
                 extension: str = ".md",
                 version: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a file to the agent's data directory.
        
        Args:
            content: The content to save
            file_type: Type of file (default: "report")
            title: Optional title for the file
            agent_name: Optional name of the agent
            extension: File extension (default: .md)
            version: Optional version identifier
            metadata: Optional metadata to save alongside the content
            
        Returns:
            The filepath of the saved file
        """
        logger.info(f"FileManager.save_file called with file_type={file_type}, title={title}")
        
        # Get human-readable timestamp
        timestamp = get_timestamp()
        
        # Clean title for filename
        if title:
            safe_title = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in title)
            safe_title = safe_title.replace(' ', '_')
            filename = f"{timestamp}_{safe_title}"
        else:
            filename = f"{timestamp}_{file_type}"
        
        # Add version if provided
        if version:
            filename = f"{filename}_v{version}"
        
        # Complete filename with extension
        if not extension.startswith('.'):
            extension = f".{extension}"
        filename = f"{filename}{extension}"
        
        # Get the appropriate directory using provided agent_name
        directory = self._get_directory_for_agent(agent_name, file_type)
        logger.info(f"Directory for saving: {directory}")
        logger.info(f"Directory exists: {os.path.exists(directory)}")
        
        os.makedirs(directory, exist_ok=True)
        
        filepath = directory / filename
        logger.info(f"Full filepath: {filepath}")
        
        # If metadata is provided, save it alongside the content
        if metadata:
            # For markdown files, we'll save metadata as JSON in a separate file
            if extension.lower() in ['.md', '.txt']:
                metadata_filepath = f"{filepath}.metadata.json"
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Metadata saved to: {metadata_filepath}")
            # For JSON files, we can include metadata in the same file
            elif extension.lower() == '.json':
                try:
                    # Try to parse the content as JSON
                    content_json = json.loads(content)
                    # Add metadata field to the JSON content
                    content_json['__metadata__'] = metadata
                    # Update the content
                    content = json.dumps(content_json, indent=2)
                except json.JSONDecodeError:
                    # If content isn't valid JSON, create metadata file instead
                    metadata_filepath = f"{filepath}.metadata.json"
                    with open(metadata_filepath, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"Content was not valid JSON. Metadata saved to separate file: {metadata_filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"File saved successfully to: {filepath}")
        
        return str(filepath)
    
    def get_file(self, file_identifier: str) -> Dict[str, Any]:
        """Get a file by its filepath.
        
        Args:
            file_identifier: The filepath of the file
            
        Returns:
            Dict containing file content and metadata
        """
        if not os.path.exists(file_identifier):
            raise FileNotFoundError(f"File not found: {file_identifier}")
        
        with open(file_identifier, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if we have metadata for this file
        metadata = None
        metadata_filepath = f"{file_identifier}.metadata.json"
        
        if os.path.exists(metadata_filepath):
            # Load metadata from separate file
            try:
                with open(metadata_filepath, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_filepath}")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata file: {metadata_filepath}")
        
        # If no separate metadata file and it's a JSON file, check for embedded metadata
        elif file_identifier.lower().endswith('.json'):
            try:
                content_json = json.loads(content)
                if '__metadata__' in content_json:
                    metadata = content_json.pop('__metadata__')
                    # Update content without the metadata
                    content = json.dumps(content_json, indent=2)
                    logger.info(f"Extracted embedded metadata from JSON: {file_identifier}")
            except json.JSONDecodeError:
                logger.debug(f"File has .json extension but content isn't valid JSON: {file_identifier}")
        
        stat = os.stat(file_identifier)
        result = {
            "content": content,
            "filepath": file_identifier,
            "created_at": stat.st_ctime
        }
        
        if metadata:
            result["metadata"] = metadata
            
        return result
    
    def list_files(self, 
                   file_type: Optional[str] = None, 
                   project_id: Optional[str] = None,
                   filter_criteria: Optional[Dict[str, Any]] = None,
                   include_metadata: bool = False) -> List[Dict[str, Any]]:
        """List files with optional filtering.
        
        Args:
            file_type: Optional filter by file type
            project_id: Optional filter by project ID
            filter_criteria: Optional additional filter criteria
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of file metadata matching the criteria
        """
        filter_criteria = filter_criteria or {}
        agent_name = filter_criteria.get("agent_name")
        results = []
        dirs_to_search = []
        if agent_name:
            if agent_name in AGENT_DIRS:
                dirs_to_search.append(AGENT_DIRS[agent_name])
            else:
                return []
        else:
            dirs_to_search = list(AGENT_DIRS.values())
        for directory in dirs_to_search:
            if os.path.exists(directory):
                for entry in os.listdir(directory):
                    full_path = os.path.join(directory, entry)
                    # Skip metadata files as they'll be handled with their primary files
                    if entry.endswith('.metadata.json'):
                        continue
                    if os.path.isfile(full_path):
                        stat = os.stat(full_path)
                        file_info = {
                            "file_id": full_path,
                            "filename": entry,
                            "created_at": stat.st_ctime
                        }
                        
                        if include_metadata:
                            # Check for metadata file
                            metadata_path = f"{full_path}.metadata.json"
                            if os.path.exists(metadata_path):
                                try:
                                    with open(metadata_path, 'r', encoding='utf-8') as f:
                                        file_info["metadata"] = json.load(f)
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse metadata file: {metadata_path}")
                            # For JSON files, check for embedded metadata
                            elif full_path.lower().endswith('.json'):
                                try:
                                    with open(full_path, 'r', encoding='utf-8') as f:
                                        content_json = json.load(f)
                                        if '__metadata__' in content_json:
                                            file_info["metadata"] = content_json['__metadata__']
                                except json.JSONDecodeError:
                                    pass
                                except Exception as e:
                                    logger.warning(f"Error reading JSON file {full_path}: {str(e)}")
                        
                        results.append(file_info)
        results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return results
    
    def delete_file(self, file_identifier: str) -> bool:
        """Delete a file by its filepath.
        
        Args:
            file_identifier: The filepath of the file
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(file_identifier):
            os.remove(file_identifier)
            return True
        return False
    
    def search_files(self, query: str, agent_name: Optional[str] = None, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Search for files containing the query in title or content.
        
        Args:
            query: Search query
            agent_name: Optional filter by agent name
            include_metadata: Whether to include metadata in the results
            
        Returns:
            List of file metadata matching the query
        """
        query = query.lower()
        results = []
        dirs_to_search = []
        if agent_name:
            if agent_name in AGENT_DIRS:
                dirs_to_search.append(AGENT_DIRS[agent_name])
            else:
                return []
        else:
            dirs_to_search = list(AGENT_DIRS.values())
        for directory in dirs_to_search:
            if os.path.exists(directory):
                for entry in os.listdir(directory):
                    # Skip metadata files as they'll be handled with their primary files
                    if entry.endswith('.metadata.json'):
                        continue
                    
                    full_path = os.path.join(directory, entry)
                    if os.path.isfile(full_path):
                        file_info = {"file_id": full_path, "filename": entry}
                        found = False
                        
                        # Check filename
                        if query in entry.lower():
                            found = True
                        
                        # Check content
                        if not found:
                            try:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    content = f.read().lower()
                                    if query in content:
                                        found = True
                            except Exception:
                                continue
                        
                        if found:
                            if include_metadata:
                                # Check for metadata file
                                metadata_path = f"{full_path}.metadata.json"
                                if os.path.exists(metadata_path):
                                    try:
                                        with open(metadata_path, 'r', encoding='utf-8') as f:
                                            file_info["metadata"] = json.load(f)
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse metadata file: {metadata_path}")
                                # For JSON files, check for embedded metadata
                                elif full_path.lower().endswith('.json'):
                                    try:
                                        with open(full_path, 'r', encoding='utf-8') as f:
                                            content_json = json.load(f)
                                            if '__metadata__' in content_json:
                                                file_info["metadata"] = content_json['__metadata__']
                                    except json.JSONDecodeError:
                                        pass
                                    except Exception as e:
                                        logger.warning(f"Error reading JSON file {full_path}: {str(e)}")
                            
                            results.append(file_info)
        return results
    
    def delete_agent_files(self, agent_type: str) -> int:
        """Delete all files for a specific agent.
        
        Args:
            agent_type: The type of agent (e.g., "researcher", "editor")
            
        Returns:
            Number of files deleted
        """
        agent_name = f"{agent_type}_agent"
        if agent_name not in AGENT_DIRS:
            print(f"Error: Invalid agent_type '{agent_type}'")
            return 0
        directory = AGENT_DIRS[agent_name]
        deleted_count = 0
        if os.path.exists(directory):
            for entry in os.listdir(directory):
                full_path = os.path.join(directory, entry)
                if os.path.isfile(full_path):
                    if self.delete_file(full_path):
                        deleted_count += 1
                        print(f"Deleted file: {full_path}")
        return deleted_count
    
    def delete_all_files(self) -> int:
        """Delete all files in all agent directories.
        
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        for directory in AGENT_DIRS.values():
            if os.path.exists(directory):
                for entry in os.listdir(directory):
                    full_path = os.path.join(directory, entry)
                    if os.path.isfile(full_path):
                        if self.delete_file(full_path):
                            deleted_count += 1
                            print(f"Deleted file: {full_path}")
        return deleted_count

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="File Manager CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List files command
    list_parser = subparsers.add_parser("list", help="List files")
    list_parser.add_argument("--agent", type=str, help="Filter by agent type (researcher, manager, editor, writer_critic, qaqc)")
    
    # Delete files command
    delete_parser = subparsers.add_parser("delete", help="Delete files")
    delete_parser.add_argument("--agent", type=str, help="Delete files for a specific agent type")
    delete_parser.add_argument("--file-id", type=str, help="Delete a specific file by ID")
    delete_parser.add_argument("--all", action="store_true", help="Delete all files")
    
    return parser.parse_args()

def main():
    """Main entry point for the File Manager CLI."""
    args = parse_args()
    
    # Initialize the file manager
    file_manager = FileManager()
    
    if args.command == "list":
        # List files
        if args.agent:
            agent_name = f"{args.agent}_agent"
            if agent_name not in AGENT_DIRS:
                print(f"Error: Invalid agent type '{args.agent}'")
                return
            
            filter_criteria = {"agent_name": agent_name}
            files = file_manager.list_files(filter_criteria=filter_criteria)
        else:
            files = file_manager.list_files()
        
        # Print the files
        print(f"Found {len(files)} files:")
        for file in files:
            print(f"- {file['filepath']} (ID: {file['file_id']})")
    
    elif args.command == "delete":
        if args.file_id:
            # Delete a specific file
            if file_manager.delete_file(args.file_id):
                print(f"Deleted file with ID: {args.file_id}")
            else:
                print(f"Error: File with ID '{args.file_id}' not found")
        
        elif args.agent:
            # Delete files for a specific agent
            deleted_count = file_manager.delete_agent_files(args.agent)
            print(f"Deleted {deleted_count} files for agent: {args.agent}")
        
        elif args.all:
            # Delete all files
            deleted_count = file_manager.delete_all_files()
            print(f"Deleted {deleted_count} files")
        
        else:
            print("Error: Please specify --file-id, --agent, or --all")
    
    else:
        print("Error: Please specify a command (list, delete)")

# This is the key change to fix the module execution issue
if __name__ == "__main__" or (
    # This handles the case when the module is run with python -m
    hasattr(sys, 'argv') and 
    len(sys.argv) > 0 and 
    'utils.file_manager.file_manager' in sys.argv[0]
):
    main() 