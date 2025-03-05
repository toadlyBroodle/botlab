import os
import yaml
import time
from datetime import datetime
import random
import logging
from typing import Dict, Any, Tuple
from smolagents import tool
from smolagents import DuckDuckGoSearchTool
from smolagents import VisitWebpageTool
import requests
import json
from pathlib import Path
import pytz

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global variables to track search state for rate limiting
_last_search_time = 0
_consecutive_failures = 0
_base_wait_time = 5.0
_current_rate_limit = _base_wait_time  # Initialize with base wait time
_max_backoff_time = 300.0 # 5 minutes

def get_timestamp() -> str:
    """Get current timestamp in a human-readable format.
    
    Returns:
        A string with the current date and time in YYYY-MM-DD_HH-MM format.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def apply_custom_agent_prompts(agent, custom_system_prompt: str = None) -> None:
    """Load and apply custom agent templates based on the agent type.
    
    Args:
        agent: The agent instance
        custom_system_prompt: Optional custom system prompt to append to the base system prompt
        
    Raises:
        ValueError: If the agent type cannot be determined or is invalid
        FileNotFoundError: If the template file is not found
        yaml.YAMLError: If there's an error parsing the YAML file
    """
    # Determine agent type from the agent's class
    agent_class_name = agent.__class__.__name__
    
    if 'CodeAgent' in agent_class_name:
        agent_type = 'code'
    elif 'ToolCallingAgent' in agent_class_name:
        agent_type = 'toolcalling'
    else:
        raise ValueError(f"Cannot determine agent type from class: {agent_class_name}")
    
    # Determine template path based on agent type
    base_dir = 'utils/agents'
    if agent_type.lower() == 'code':
        template_path = os.path.join(base_dir, 'code_agent.yaml')
    elif agent_type.lower() == 'toolcalling':
        template_path = os.path.join(base_dir, 'toolcalling_agent.yaml')
    else:
        raise ValueError(f"Invalid agent type: {agent_type}. Must be 'code' or 'toolcalling'")
    
    # Load the template
    try:
        with open(template_path, 'r') as file:
            template = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    
    # Apply the template to the agent's prompt_templates
    agent.prompt_templates = template
    
    # append todays date and time to all system prompts
    agent.prompt_templates["system_prompt"] += f"\n\nToday's date and time is {datetime.now().strftime('%Y-%m-%d %H:%M')}."
    
    # Append custom system prompt if provided
    if custom_system_prompt:
        agent.prompt_templates["system_prompt"] += f"\n\n{custom_system_prompt}"
    
    # Reinitialize the system prompt to apply the new template
    agent.system_prompt = agent.initialize_system_prompt()

@tool
def web_search(query: str, max_results: int = 10, rate_limit_seconds: float = 5.0, max_retries: int = 3) -> str:
    """Performs a DuckDuckGo web search with intelligent rate limiting to avoid being blocked.
    
    This tool uses exponential backoff and jitter to handle rate limits gracefully. When rate limits
    are encountered, it will automatically retry with increasing wait times between attempts.
    
    DuckDuckGo search operators:
    - Use quotes for exact phrases: "climate change solutions"
    - Use '-' to exclude terms: climate -politics
    - Use '+' to emphasize terms: climate +solutions
    - Use 'filetype:' for specific file types: "research paper filetype:pdf"
    - Use 'site:' to search specific websites: "AI advances site:stanford.edu"
    - Use '-site:' to exclude websites: "AI breakthroughs -site:wikipedia.org"
    - Use 'intitle:' to search in page titles: intitle:climate
    - Use 'inurl:' to search in URLs: inurl:research
    - Use '~' for related terms: ~"artificial intelligence"
    
    Args:
        query: The search query to perform. Can include special operators like site: or filetype:
        max_results: Maximum number of results to return (default: 10)
        rate_limit_seconds: Minimum seconds to wait between searches (default: 5.0)
        max_retries: Maximum number of retry attempts when rate limited (default: 5)
        
    Returns:
        Formatted search results as a markdown string with titles, URLs, and snippets
    """
    global _last_search_time, _consecutive_failures, _base_wait_time, _max_backoff_time, _current_rate_limit

    if rate_limit_seconds < 5.0:
        rate_limit_seconds = 5.0
    
    # Calculate current wait time with aggressive exponential backoff
    current_wait_time = rate_limit_seconds * (3 ** _consecutive_failures)
    
    # Add jitter to avoid synchronized requests (±30%)
    current_wait_time = current_wait_time * (0.7 + 0.6 * random.random())
    
    # Cap the wait time at the maximum backoff time
    current_wait_time = min(current_wait_time, _max_backoff_time)
    
    # Store the current rate limit
    _current_rate_limit = current_wait_time
    
    # Apply rate limiting
    current_time = time.time()
    time_since_last_search = current_time - _last_search_time
    
    if time_since_last_search < current_wait_time:
        # Wait the remaining time to respect the rate limit
        sleep_time = current_wait_time - time_since_last_search
        logger.info(f"Waiting {sleep_time:.2f} seconds to respect DDGS rate limits (failure count: {_consecutive_failures})")
        time.sleep(sleep_time)
    
    # Create a DuckDuckGoSearchTool instance
    search_tool = DuckDuckGoSearchTool(max_results=max_results)
    
    # Try to perform the search with retries
    retries = 0
    while retries <= max_retries:
        try:
            result = search_tool.forward(query)
            
            # Check if the result is empty or contains an error message
            if not result or "Error" in result:
                # This might be a rate limit that didn't raise an exception
                _consecutive_failures += 1
                retries += 1
                
                if retries <= max_retries:
                    # Calculate backoff time with more aggressive multiplier
                    backoff_time = _base_wait_time * (3 ** _consecutive_failures)
                    # Add jitter (±30%)
                    backoff_time = backoff_time * (0.7 + 0.6 * random.random())
                    # Cap the backoff time
                    backoff_time = min(backoff_time, _max_backoff_time)
                    # Update current rate limit
                    _current_rate_limit = backoff_time
                    
                    logger.info(f"Empty result or error detected: {result}.\n Retrying in {backoff_time:.2f} seconds (failure count: {_consecutive_failures})...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return f"Error: DuckDuckGo search failed after {max_retries} retries. Last result: {result}"
            
            # Success - reset consecutive failures, rate limit, and update last search time
            _consecutive_failures = 0
            _current_rate_limit = _base_wait_time  # Reset rate limit back to base
            _last_search_time = time.time()
            
            logger.info(f"Search successful. Rate limit reset to base value: {_base_wait_time} seconds.")
            return result
            
        except Exception as e:
            error_message = str(e)
            _last_search_time = time.time()
            
            # Check if it's a rate limit error - look for various indicators
            if any(term in error_message.lower() for term in ["ratelimit", "rate limit", "429", "too many requests", "202", "blocked", "forbidden", "403"]):
                _consecutive_failures += 1
                retries += 1
                
                if retries <= max_retries:
                    # Calculate backoff time with increased base wait time and more aggressive multiplier
                    backoff_time = _base_wait_time * (3 ** _consecutive_failures)
                    # Add jitter (±30%)
                    backoff_time = backoff_time * (0.7 + 0.6 * random.random())
                    # Cap the backoff time
                    backoff_time = min(backoff_time, _max_backoff_time)
                    # Update current rate limit
                    _current_rate_limit = backoff_time
                    
                    logger.info(f"Rate limit detected. Retrying in {backoff_time:.2f} seconds (failure count: {_consecutive_failures})...")
                    time.sleep(backoff_time)
                else:
                    # Max retries exceeded
                    return f"Error: DuckDuckGo search rate limit exceeded after {max_retries} retries."
            else:
                # Not a rate limit error, just return the error
                return f"Error performing search: {error_message}"
    
    # This should not be reached, but just in case
    return "Error: Unable to complete search after multiple attempts."

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    # Use the VisitWebpageTool from smolagents
    webpage_tool = VisitWebpageTool()
    return webpage_tool.forward(url)

@tool
def load_latest_draft(agent_name: str = None) -> str:
    """Load the most recent draft file.
    
    This tool retrieves the most recent draft file from the system. Draft files are typically saved by:
    - Writer agent: When creating draft versions
    - Editor agent: When saving major revisions
    
    The drafts are stored in chronological order, so this will return the most recently saved draft
    by default. If agent_name is specified, it will return the most recent draft created by that agent.
    
    Use this tool to continue work from where a previous agent left off. The tool will show which 
    agent created the draft in the "Created by" field.
    
    Args:
        agent_name: Optional. If provided, only drafts created by this agent will be considered.
                   Common values: "writer_agent", "editor_agent"
    
    Returns:
        The content of the most recent draft, or an error message if no drafts are found.
    """
    try:
        from utils.file_manager import FileManager
        from datetime import datetime
        
        # Initialize file manager
        file_manager = FileManager()
        
        # Get all draft files
        drafts = file_manager.list_files(file_type="draft")
        
        if not drafts:
            return "No draft files found."
        
        # Filter by agent_name if specified
        if agent_name:
            filtered_drafts = []
            for draft in drafts:
                # Get the full file to check metadata
                full_draft = file_manager.get_file(draft['file_id'])
                draft_agent = full_draft["metadata"].get("agent_name", 
                              full_draft["metadata"].get("source", "Unknown"))
                
                if draft_agent == agent_name:
                    filtered_drafts.append(draft)
            
            if not filtered_drafts:
                return f"No draft files found created by '{agent_name}'."
            
            drafts = filtered_drafts
        
        # Sort by creation date (newest first)
        sorted_drafts = sorted(drafts, key=lambda x: x.get('created_at', ''), reverse=True)
        
        if sorted_drafts:
            # Get the latest draft
            latest_draft = file_manager.get_file(sorted_drafts[0]['file_id'])
            
            # Return information about the draft
            title = latest_draft["metadata"].get("title", "Untitled")
            created_at = latest_draft["metadata"].get("created_at", "Unknown date")
            word_count = latest_draft["metadata"].get("word_count", "Unknown")
            
            # Get agent information
            source = latest_draft["metadata"].get("source", "Unknown")
            agent_name = latest_draft["metadata"].get("agent_name", source)
            
            # Format creation date for better readability
            try:
                created_dt = datetime.fromisoformat(created_at)
                created_at = created_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass  # Keep the original format if parsing fails
            
            return f"Latest draft: '{title}'\nCreated: {created_at}\nWords: {word_count}\nCreated by: {agent_name}\n\n{latest_draft['content']}"
        
        return "No draft files found."
        
    except Exception as e:
        return f"Error loading latest draft: {str(e)}"

@tool
def load_latest_report(agent_name: str = None) -> str:
    """Load the most recent report file.
    
    This tool retrieves the most recent report file from the system. Report files are typically saved by:
    - Researcher agent: When saving research findings and compiled information
    - Editor agent: When saving final, polished content after thorough review and fact-checking
    
    The reports are stored in chronological order, so this will return the most recently saved report
    by default. If agent_name is specified, it will return the most recent report created by that agent.
    
    Use this tool to access the most recent finalized content or research. The tool will show which 
    agent created the report in the "Created by" field.
    
    Args:
        agent_name: Optional. If provided, only reports created by this agent will be considered.
                   Common values: "researcher_agent", "editor_agent"
    
    Returns:
        The content of the most recent report, or an error message if no reports are found.
    """
    try:
        from utils.file_manager import FileManager
        from datetime import datetime
        
        # Initialize file manager
        file_manager = FileManager()
        
        # Get all report files
        reports = file_manager.list_files(file_type="report")
        
        if not reports:
            return "No report files found."
        
        # Filter by agent_name if specified
        if agent_name:
            filtered_reports = []
            for report in reports:
                # Get the full file to check metadata
                full_report = file_manager.get_file(report['file_id'])
                report_agent = full_report["metadata"].get("agent_name", 
                               full_report["metadata"].get("source", "Unknown"))
                
                if report_agent == agent_name:
                    filtered_reports.append(report)
            
            if not filtered_reports:
                return f"No report files found created by '{agent_name}'."
            
            reports = filtered_reports
        
        # Sort by creation date (newest first)
        sorted_reports = sorted(reports, key=lambda x: x.get('created_at', ''), reverse=True)
        
        if sorted_reports:
            # Get the latest report
            latest_report = file_manager.get_file(sorted_reports[0]['file_id'])
            
            # Return information about the report
            title = latest_report["metadata"].get("title", "Untitled")
            created_at = latest_report["metadata"].get("created_at", "Unknown date")
            word_count = latest_report["metadata"].get("word_count", "Unknown")
            
            # Get agent information
            source = latest_report["metadata"].get("source", "Unknown")
            agent_name = latest_report["metadata"].get("agent_name", source)
            
            # Format creation date for better readability
            try:
                created_dt = datetime.fromisoformat(created_at)
                created_at = created_dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass  # Keep the original format if parsing fails
            
            return f"Latest report: '{title}'\nCreated: {created_at}\nWords: {word_count}\nCreated by: {agent_name}\n\n{latest_report['content']}"
        
        return "No report files found."
        
    except Exception as e:
        return f"Error loading latest report: {str(e)}"

@tool
def load_file(file_identifier: str) -> str:
    """Load a file by its unique ID or filename.
    
    This tool retrieves a specific file using either its unique identifier or filename. 
    File IDs are typically returned when an agent saves a file, while filenames include 
    the timestamp and title in the format: "YYYYMMDD_HHMMSS_projectid_title.md"
    
    Use this tool when you need to access a specific file that isn't the latest. The tool
    will show detailed metadata including which agent created the file.
    
    The file types and their typical sources are:
    
    - draft: Writer agent drafts and Editor agent edits
    - report: Researcher agent findings and Editor agent final content
    - paper: Research papers downloaded and converted by the Researcher agent
    - resource: Various downloaded resources and cached data
    
    Args:
        file_identifier: The unique identifier or filename of the file
        
    Returns:
        The content of the file, or an error message if the file is not found.
    """
    try:
        from utils.file_manager import FileManager
        from datetime import datetime
        
        # Initialize file manager
        file_manager = FileManager()
        
        # Get the file
        file_data = file_manager.get_file(file_identifier)
        
        # Return information about the file
        title = file_data["metadata"].get("title", "Untitled")
        file_type = file_data["metadata"].get("file_type", "Unknown type")
        created_at = file_data["metadata"].get("created_at", "Unknown date")
        
        # Get agent information
        source = file_data["metadata"].get("source", "Unknown")
        agent_name = file_data["metadata"].get("agent_name", source)
        
        # Format creation date for better readability
        try:
            created_dt = datetime.fromisoformat(created_at)
            created_at = created_dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass  # Keep the original format if parsing fails
        
        # Get additional metadata that might be useful
        word_count = file_data["metadata"].get("word_count", "Unknown")
        version = file_data["metadata"].get("version", "")
        version_info = f", Version: {version}" if version else ""
        
        return f"File: '{title}'\nType: {file_type}\nCreated: {created_at}{version_info}\nWords: {word_count}\nCreated by: {agent_name}\n\n{file_data['content']}"
        
    except Exception as e:
        return f"Error loading file '{file_identifier}': {str(e)}"

