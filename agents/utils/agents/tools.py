import os
import yaml
import time
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, Any, Tuple, Optional, Union
from smolagents import tool
from smolagents import DuckDuckGoSearchTool
from smolagents import VisitWebpageTool
import requests
import json
from pathlib import Path
import pytz
import re
import uuid
import hashlib
from smolagents import Tool
from ..logger_config import setup_logger

# Import Google Generative AI library for direct Gemini search
from google import genai
from google.genai.types import Tool as GenaiTool, GenerateContentConfig, GoogleSearch
from PIL import Image
from io import BytesIO
import base64

# Added for user feedback tools
import subprocess
import mailbox
from dotenv import load_dotenv

# Load environment variables for user feedback tools
load_dotenv()

# Get botlab root directory
BOTLAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

LOGS_DIR = os.path.join(BOTLAB_ROOT, 'agents', 'logs')

# Set up logging with dedicated file in appropriate logs directory
logger = setup_logger('botlab_tools', log_dir=LOGS_DIR)

# Global variables to track search state for rate limiting
_last_search_time = 0
_consecutive_failures = 0
_base_wait_time = 5.0
_current_rate_limit = _base_wait_time  # Initialize with base wait time
_max_backoff_time = 300.0 # 5 minutes

# Variables for Gemini search fallback
_using_gemini_fallback = False
_gemini_fallback_until = 0  # Timestamp until which to use Gemini fallback
_gemini_fallback_duration = 300  # Duration to use Gemini fallback in seconds (5 minutes)
_gemini_search_count = 0  # Count of Gemini searches performed today
_gemini_search_limit = 500  # Daily limit for Gemini searches
_gemini_search_reset_time = 0  # Time when the search count was last reset
_gemini_client = None  # Lazy-loaded Gemini client

# Constants for user feedback tools
DEFAULT_MAILBOX_PATH = "/home/fb_agent/var/mail"  # Corrected Maildir path
COMMAND_PATTERNS = {
    "frequency": r"FREQUENCY:\s*(\S+.*?)(?=\n\w+:|$)", # General pattern, specific parsing later
    "detail": r"DETAIL:\s*(\S+.*?)(?=\n\w+:|$)",
    "focus": r"FOCUS:\s*(\S+.*?)(?=\n\w+:|$)",
    "feedback": r"FEEDBACK:\s*(.*?)(?=\n\w+:|$)", # Feedback can be multi-line
    "pause": r"PAUSE:\s*(\S+.*?)(?=\n\w+:|$)",
    "resume": r"RESUME:\s*(\S+.*?)(?=\n\w+:|$)"
}

COMMAND_TEMPLATE_PLACEHOLDERS = {
    "frequency": "[Enter number, e.g., 1 for every iteration, 5 for every 5 iterations]",
    "detail": "[low|medium|high]",
    "focus": "[Enter focus area, e.g., research, writing, specific_feature]",
    "feedback": "[Provide any specific feedback or instructions here]",
    "pause": "[true|false]",
    "resume": "[true|false]"
}

EMAIL_COMMAND_TEMPLATE = """
------------------------------------
COMMANDS - Reply to this email with updated values below:
(Remove or leave blank any commands you don't want to change)

FREQUENCY: {frequency}
DETAIL: {detail}
FOCUS: {focus}
FEEDBACK: {feedback}
PAUSE: {pause}
RESUME: {resume}
------------------------------------
""".format(**COMMAND_TEMPLATE_PLACEHOLDERS)

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
    if not hasattr(agent, 'name'):
        raise ValueError("Agent must have a 'name' attribute to determine its type")
    
    # Apply the custom system prompt if provided
    if custom_system_prompt:
        agent.system_prompt = custom_system_prompt
    
    # Determine agent type from the agent's class
    agent_class_name = agent.__class__.__name__
    
    if 'CodeAgent' in agent_class_name:
        agent_type = 'code'
    elif 'ToolCallingAgent' in agent_class_name:
        agent_type = 'toolcalling'
    else:
        raise ValueError(f"Cannot determine agent type from class: {agent_class_name}")
    
    # Determine template path based on agent type
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
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

def _initialize_gemini_client():
    """Initialize the Gemini client for search.
    
    Returns:
        The initialized Gemini client or None if initialization fails
    """
    global _gemini_client
    
    if _gemini_client is None:
        # Get the Google API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini search fallback will not work.")
            return None
            
        try:
            # Initialize the Gemini client with the current API
            # The Client constructor now takes the API key directly
            _gemini_client = genai.Client(api_key=api_key)
            logger.info("Initialized Gemini client for search fallback")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return None
            
    return _gemini_client

def _check_gemini_search_limit():
    """Check if we've exceeded the daily Gemini search limit.
    
    Returns:
        Tuple of (limit_exceeded, current_count, limit)
    """
    global _gemini_search_count, _gemini_search_limit, _gemini_search_reset_time
    
    # Check if we need to reset the counter (daily)
    current_time = time.time()
    if _gemini_search_reset_time == 0:
        # Initialize reset time to next midnight
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        _gemini_search_reset_time = tomorrow.timestamp()
    elif current_time > _gemini_search_reset_time:
        # Reset counter and set new reset time
        _gemini_search_count = 0
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        _gemini_search_reset_time = tomorrow.timestamp()
        logger.info(f"Reset Gemini search counter. Next reset at {datetime.fromtimestamp(_gemini_search_reset_time)}")
    
    # Check if we've exceeded the limit
    return _gemini_search_count >= _gemini_search_limit, _gemini_search_count, _gemini_search_limit

def _perform_gemini_search(query: str, max_results: int = 10) -> str:
    """Perform a search using Gemini's search grounding capability.
    
    Args:
        query: The search query
        max_results: Maximum number of results (not directly used but kept for API compatibility)
        
    Returns:
        Formatted search results as a markdown string
    """
    global _gemini_search_count
    
    # Check if we've exceeded the daily search limit
    limit_exceeded, current_count, limit = _check_gemini_search_limit()
    if limit_exceeded:
        return f"Error: Daily Gemini search limit exceeded ({current_count}/{limit}). Try again tomorrow."
    
    # Initialize the Gemini client
    client = _initialize_gemini_client()
    if client is None:
        return "Error: Gemini search unavailable. GEMINI_API_KEY may not be set."
    
    try:
        # Create the search tool
        google_search_tool = GenaiTool(
            google_search=GoogleSearch()
        )
        
        # Perform the search using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        
        # Increment the search counter
        _gemini_search_count += 1
        
        # Extract the main content
        main_content = ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'text'):
                        main_content += part.text + "\n"
        
        # Format the result
        result = f"# Search Results for: {query}\n\n{main_content.strip()}\n\n"
        
        # Add grounding sources if available
        sources_info = ""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_metadata = candidate.grounding_metadata
                
                # Add search suggestions if available
                if hasattr(grounding_metadata, 'web_search_queries') and grounding_metadata.web_search_queries:
                    sources_info += "\n## Related Searches\n"
                    for query in grounding_metadata.web_search_queries:
                        sources_info += f"- {query}\n"
                
                # Add source information if available
                if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
                    sources_info += "\n## Sources\n"
                    for i, chunk in enumerate(grounding_metadata.grounding_chunks):
                        if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri') and hasattr(chunk.web, 'title'):
                            sources_info += f"{i+1}. [{chunk.web.title}]({chunk.web.uri})\n"
        
        # Add sources information if available
        if sources_info:
            result += sources_info
        
        # Add a note about the search provider
        result += f"\n\n---\n*Note: These results were provided by Gemini Search due to DuckDuckGo rate limiting. Search count: {_gemini_search_count}/{_gemini_search_limit}*"
        
        return result
    except Exception as e:
        logger.error(f"Gemini search error: {e}")
        return f"Error performing Gemini search: {str(e)}"

@tool
def web_search(query: str, max_results: int = 10, rate_limit_seconds: float = 5.0, max_retries: int = 3) -> str:
    """Performs a web search with intelligent rate limiting and fallback mechanisms.
    
    This tool primarily uses DuckDuckGo for web searches, but will temporarily switch to
    Gemini Search if DuckDuckGo rate limits are encountered repeatedly. This provides
    a robust search capability that can handle rate limiting gracefully.
    
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
        max_retries: Maximum number of retry attempts when rate limited (default: 3, max allowed: 5)
        
    Returns:
        Formatted search results as a markdown string with titles, URLs, and snippets
    """
    global _last_search_time, _consecutive_failures, _base_wait_time, _max_backoff_time, _current_rate_limit
    global _using_gemini_fallback, _gemini_fallback_until, _gemini_fallback_duration

    # Check if we should use Gemini fallback
    current_time = time.time()
    if _using_gemini_fallback and current_time < _gemini_fallback_until:
        # We're in fallback mode, use Gemini search
        time_left = int(_gemini_fallback_until - current_time)
        logger.info(f"Using Gemini search fallback (DuckDuckGo cooling off for {time_left} more seconds)")
        return _perform_gemini_search(query, max_results)

    # If fallback period has expired, reset the fallback flag
    if _using_gemini_fallback and current_time >= _gemini_fallback_until:
        _using_gemini_fallback = False
        _consecutive_failures = 0  # Reset failures when coming out of fallback
        logger.info("Fallback period expired, switching back to DuckDuckGo search")

    # Limit max_retries to 5 if a larger value is passed
    if max_retries > 5:
        max_retries = 5

    # Reset consecutive failures if it's already at or above the max_retries limit
    # This prevents the function from carrying over too many failures from previous calls
    if _consecutive_failures >= max_retries:
        _consecutive_failures = 0

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
                
                if retries <= max_retries and _consecutive_failures <= max_retries:
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
                    # Max retries exceeded - switch to Gemini fallback
                    logger.warning(f"DuckDuckGo search failed after {retries} retries. Switching to Gemini search fallback.")
                    _using_gemini_fallback = True
                    _gemini_fallback_until = time.time() + _gemini_fallback_duration
                    return _perform_gemini_search(query, max_results)
            
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
                
                if retries <= max_retries and _consecutive_failures <= max_retries:
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
                    # Max retries exceeded - switch to Gemini fallback
                    logger.warning(f"DuckDuckGo search rate limited after {retries} retries. Switching to Gemini search fallback.")
                    _using_gemini_fallback = True
                    _gemini_fallback_until = time.time() + _gemini_fallback_duration
                    return _perform_gemini_search(query, max_results)
            else:
                # Not a rate limit error, just return the error
                logger.error(f"Non-rate-limit error in search: {error_message}")
                return f"Error performing search: {error_message}"
    
    # This should not be reached, but just in case
    # Switch to Gemini fallback if we've had too many failures
    logger.warning("Unable to complete search after multiple attempts. Switching to Gemini search fallback.")
    _using_gemini_fallback = True
    _gemini_fallback_until = time.time() + _gemini_fallback_duration
    return _perform_gemini_search(query, max_results)

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
def load_file(agent_type: Optional[str] = None, version: str = "latest") -> str:
    """Load a file from an agent's data directory.
    
    This tool retrieves files from agent-specific data directories. Each agent stores its
    output files in its own data directory (e.g., researcher/data/, editor/data/).
    
    Args:
        agent_type: The type of agent to load files from. Options include:
                   "researcher", "manager", "editor", "writer_critic", "qaqc".
                   If None, files from all agents will be considered.
        version: Which version to load. Options:
                - "latest": The most recent file (default)
                - "previous": The second most recent file
    
    Returns:
        The content of the file, or an error message if no files are found.
    """
    # Import here to avoid circular imports
    from ...utils.file_manager.file_manager import FileManager, AGENT_DIRS
    
    try:
        import logging
        logger = logging.getLogger("file_manager")
        
        # Initialize file manager
        file_manager = FileManager()
        
        # Get list of files from the agent's directory
        filter_criteria = {}
        if agent_type:
            filter_criteria["agent_name"] = f"{agent_type}_agent"
            
        files = file_manager.list_files(filter_criteria=filter_criteria)
        
        if not files:
            agent_list = ", ".join([a.replace("_agent", "") for a in AGENT_DIRS.keys()])
            return f"No files found for agent_type={agent_type}. Valid agent types: {agent_list}"
        
        # Sort files by creation time (newest first)
        sorted_files = sorted(files, key=lambda x: x.get("created_at", 0), reverse=True)
        
        # Get the requested version
        if version == "latest":
            index = 0
        elif version == "previous":
            if len(sorted_files) < 2:
                return "No previous version found. Only one file exists."
            index = 1
        else:
            return f"Error: Invalid version '{version}'. Valid options are: 'latest' or 'previous'."
        
        # Get the file data
        file_data = file_manager.get_file(sorted_files[index]["file_id"])
        return file_data["content"]
        
    except Exception as e:
        import traceback
        return f"Error loading file: {str(e)}\n{traceback.format_exc()}"

def extract_final_answer_from_memory(agent) -> Any:
    """Extract the final_answer from an agent's memory.
    
    Args:
        agent: The agent instance with memory
        
    Returns:
        The final answer content if found, or None if not found
    """
    # Check if the agent has memory and steps
    if hasattr(agent, 'memory') and hasattr(agent.memory, 'steps'):
        # Look for tool calls in the agent's memory steps
        for step in agent.memory.steps:
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tool_call in step.tool_calls:
                    if tool_call.name == "final_answer":
                        # Extract the final answer content
                        return tool_call.result
    
    return None

def save_final_answer(agent, result: str, query_or_prompt: str = None, agent_type: str = "agent") -> str:
    """Save the agent's final answer to a file in the agent's data directory.
    
    This function extracts the final_answer from an agent's memory if available,
    otherwise it uses the provided result. It then saves the content to a file
    in the agent's data directory.
    
    Args:
        agent: The agent instance with memory
        result: The result from the agent.run() call
        query_or_prompt: The original query or prompt that was given to the agent
        agent_type: The type of agent (e.g., "researcher", "editor")
    
    Returns:
        The file ID of the saved file, or an error message
    """
    # Import here to avoid circular imports
    from ...utils.file_manager.file_manager import FileManager, AGENT_DIRS
    
    try:
        import os
        import logging
        logger = logging.getLogger("file_manager")
        
        # Print debug information
        logger.info(f"Saving final answer for agent_type: {agent_type}")
        
        # Extract the final answer from agent.memory if available
        final_answer = extract_final_answer_from_memory(agent)
        
        # If no final answer found, use the result as-is
        if not final_answer:
            logger.info("No final_answer found in agent memory, using result")
            final_answer = result
            
        # Construct metadata
        metadata = {
            "agent_name": f"{agent_type}_agent",
            "query": query_or_prompt if query_or_prompt else "No query provided",
            "content_type": "final_answer",
        }
        
        # Save the final answer
        file_manager = FileManager()
        file_id = file_manager.save_file(
            content=final_answer,
            metadata=metadata
        )
        
        logger.info(f"Saved final answer as file_id: {file_id}")
        return file_id
        
    except Exception as e:
        import traceback
        logger.error(f"Error saving final answer: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error saving final answer: {str(e)}"

@tool
def generate_image(prompt: str) -> str:
    """Generate an image using Google's new gemini-2.0-flash-exp-image-generation model.
    
    Args:
        prompt (str): A detailed text description of the image you want to generate.
            The more detailed and specific the prompt, the better the results.
            This tool is unable to generate sexually suggestive or harmful images, so make sure you soften any NSFW user prompts.
            
    Returns:
        str: The URL of generated image.
        
    Raises:
        Exception: If there is an error generating or saving the image.
    """
    try:
        # Check required environment variables
        images_path = os.getenv('IMAGES_SAVE_PATH')
        images_url = os.getenv('IMAGES_BASE_URL')
        if not images_path or not images_url:
            raise Exception("IMAGES_SAVE_PATH and IMAGES_BASE_URL environment variables must be set")

        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise Exception("GEMINI_API_KEY environment variable must be set")
        client = genai.Client(api_key=api_key)
        
        # Generate the image
        logger.info(f"Generating image: {prompt[:100]}...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        # Log prompt feedback and code execution result only if they contain data
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            logger.info(f"Complete prompt feedback: {response.prompt_feedback}")

        if hasattr(response, 'code_execution_result') and response.code_execution_result:
            logger.info(f"Code execution result: {response.code_execution_result}")

        # Check for response text that might indicate issues
        if hasattr(response, 'text') and response.text:
            logger.warning(f"Response included text message: {response.text}")
            
        # Check for prompt feedback (safety checks, etc)
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback = response.prompt_feedback
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                logger.error(f"Prompt blocked: {feedback.block_reason}")
                raise Exception(f"Image generation blocked: {feedback.block_reason}")
            if hasattr(feedback, 'safety_ratings') and feedback.safety_ratings:
                for rating in feedback.safety_ratings:
                    if hasattr(rating, 'probability') and hasattr(rating, 'category'):
                        if rating.probability > 3:  # High probability of issue
                            logger.warning(f"Safety concern: {rating.category} (probability: {rating.probability})")
        
        # Extract image data
        image_data = None
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    mime_type = getattr(part.inline_data, 'mime_type', None)
                    if mime_type == 'image/png':
                        try:
                            # Get the base64 data
                            base64_data = part.inline_data.data
                            
                            # If it's already bytes, use it directly
                            if isinstance(base64_data, bytes):
                                image_data = base64_data
                            else:
                                # If it's a string, decode it from base64
                                if isinstance(base64_data, str):
                                    missing_padding = len(base64_data) % 4
                                    if missing_padding:
                                        base64_data += '=' * (4 - missing_padding)
                                    image_data = base64.b64decode(base64_data)
                                else:
                                    raise Exception(f"Unexpected data type: {type(base64_data)}")
                            
                            # Verify it's a valid PNG
                            if not image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                                raise Exception("Generated data is not a valid PNG")
                                
                            # Validate image with PIL
                            with BytesIO(image_data) as bio:
                                img = Image.open(bio)
                                img.verify()
                                logger.debug(f"Image validated: {img.format} ({img.size[0]}x{img.size[1]})")
                            break
                            
                        except Exception as e:
                            logger.error(f"Error processing image data: {e}")
                            raise
        
        if not image_data:
            # Check if there's any useful error information in the response
            error_msg = "No valid image data found in response"
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    error_msg += f" (Finish reason: {candidate.finish_reason})"
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            error_msg += f" - {part.text}"
            raise Exception(error_msg)
        
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"gemini_{timestamp}_{prompt_hash}.png"
        
        # Save to the dedicated directory
        os.makedirs(images_path, exist_ok=True)
        final_path = os.path.join(images_path, filename)
        
        with open(final_path, 'wb') as f:
            f.write(image_data)
            
        file_size = os.path.getsize(final_path)
        logger.info(f"Image saved: {filename} ({file_size/1024:.1f}KB)")
        
        # Return the URL where the image will be served
        return f"{images_url.rstrip('/')}/{filename}"
        
    except Exception as e:
        logger.error(f"Failed to generate image: {str(e)}")
        raise Exception(f"Error generating/saving image: {str(e)}")

def send_mail(subject: str, body: str) -> str:
    """Send an email using the sendmail command with explicit headers and envelope sender.
    
    Args:
        subject: Subject line of the email
        body: Body content of the email
        
    Returns:
        Status message indicating success or failure
    """
    try:
        recipient = os.getenv("REMOTE_USER_EMAIL")
        sender = os.getenv("LOCAL_USER_EMAIL", "fb_agent@botlab.dev")
        
        if not recipient:
            logger.error("REMOTE_USER_EMAIL environment variable is not set. Cannot send email.")
            return "REMOTE_USER_EMAIL environment variable is not set. Cannot send email."
            
        # Append command template to the body
        full_body = f"{body}\n\n{EMAIL_COMMAND_TEMPLATE}"
            
        # Format the current time for the Date header
        timestamp = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
        if not timestamp.endswith(("+0000", "-0000")) and len(timestamp.split()) == 5:
             offset_seconds = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
             offset_hours = abs(offset_seconds) // 3600
             offset_minutes = (abs(offset_seconds) % 3600) // 60
             offset_sign = "-" if offset_seconds > 0 else "+"
             timestamp += f" {offset_sign}{offset_hours:02d}{offset_minutes:02d}" 

        # Construct the email message with RFC 5322 headers
        email_content_lines = [
            f"From: {sender}",
            f"To: {recipient}",
            f"Subject: {subject}", 
            f"Date: {timestamp}",
            f"User-Agent: BotLab Feedback Agent", 
            "", # Blank line separating headers from body
            full_body
        ]
        email_content = '\n'.join(email_content_lines)
        
        logger.debug(f"--- Email Content ---\n{email_content}\n---------------------")
        
        # Use sendmail -t (read headers) and -f (set envelope sender)
        sendmail_command = ["/usr/sbin/sendmail", "-t", "-f", sender]
        logger.info(f"Attempting to send email via command: {sendmail_command}")

        # Prepare a minimal environment, keeping only PATH
        minimal_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")}
        logger.debug(f"Using minimal environment: {minimal_env}")

        process = subprocess.Popen(sendmail_command, 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True, 
                                   encoding='utf-8',
                                   env=minimal_env) # Pass the minimal environment
                                   
        stdout, stderr = process.communicate(input=email_content)
        return_code = process.returncode
        
        # Log results
        logger.info(f"sendmail execution finished.")
        logger.info(f"Return Code: {return_code}")
        if stdout:
            logger.info(f"sendmail stdout: {stdout.strip()}")
        if stderr:
            # Log stderr as warning unless return code is non-zero
            if return_code != 0:
                logger.error(f"sendmail stderr: {stderr.strip()}")
            else:
                 logger.warning(f"sendmail stderr: {stderr.strip()}")
            
        # Check return code
        if return_code == 0:
            logger.info(f"Email command via sendmail -t -f executed successfully. Assumed sent from {sender} to {recipient}.")
            return f"Email sent successfully to {recipient}."
        else:
            error_msg = f"sendmail command failed with code {return_code}. Stderr: {stderr.strip() if stderr else 'N/A'}"
            logger.error(error_msg)
            # Also return the error message to the caller
            return error_msg 

    except Exception as e:
        error_msg = f"Unexpected Python error in send_mail: {str(e)}"
        logger.exception(error_msg) # Log full exception traceback
        return error_msg

def check_mail() -> Dict[str, Any]:
    """Check for the most recent unread email from REMOTE_USER_EMAIL in the fb_agent user's maildir.
    This function checks a Maildir-format mailbox in /home/fb_agent/var/mail/
    
    This function assumes the main application user has read access to fb_agent's maildir.
        
    Returns:
        Dictionary containing the most recent unread message details or empty dict if no unread messages
    """
    try:
        # Use the fb_agent user's maildir
        maildir_path = DEFAULT_MAILBOX_PATH
        remote_email = os.getenv("REMOTE_USER_EMAIL")
        
        if not remote_email:
            logger.error("REMOTE_USER_EMAIL not found in environment variables")
            return {}
        
        # Check if maildir exists
        new_mail_dir = os.path.join(maildir_path, "new")
        if not os.path.exists(new_mail_dir):
            logger.error(f"No maildir found at {new_mail_dir}")
            return {}
        
        # Check if we have read access to the maildir
        if not os.access(new_mail_dir, os.R_OK):
            error_msg = f"No read access to {new_mail_dir}. Make sure the current user has proper permissions."
            logger.error(error_msg)
            return {"error": error_msg}
        
        # List all files in the new mail directory
        mail_files = [f for f in os.listdir(new_mail_dir) if os.path.isfile(os.path.join(new_mail_dir, f))]
        
        if not mail_files:
            logger.info("No new mail files found in Maildir")
            return {}
        
        # Sort mail files by modification time (newest first)
        mail_files.sort(key=lambda f: os.path.getmtime(os.path.join(new_mail_dir, f)), reverse=True)
        
        # Variables to track the most recent message
        most_recent_message = None
        most_recent_file = None
        
        # Process mail files to find the first one from the target email
        for mail_file in mail_files:
            file_path = os.path.join(new_mail_dir, mail_file)
            
            try:
                # Read the mail file
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    mail_content = f.read()
                
                # Extract headers and body
                headers, _, body = mail_content.partition('\n\n')
                headers = headers.split('\n')
                
                # Extract From header
                from_addr = ""
                subject = "No Subject"
                date_str = ""
                
                for header in headers:
                    if header.startswith('From:'):
                        from_addr = header[5:].strip()
                    elif header.startswith('Subject:'):
                        subject = header[8:].strip()
                    elif header.startswith('Date:'):
                        date_str = header[5:].strip()
                
                # Check if message is from the target email
                if remote_email.lower() not in from_addr.lower():
                    continue
                
                # Found a matching email
                most_recent_message = {
                    "from": from_addr,
                    "subject": subject,
                    "date": date_str,
                    "body": body.strip()
                }
                
                most_recent_file = file_path
                logger.info(f"Found matching email: {file_path}")
                break  # Stop after finding the first matching email
                
            except Exception as e:
                logger.error(f"Error reading mail file {file_path}: {e}")
                continue
        
        # Check if we have write access to move the file to 'cur' to mark as read
        cur_mail_dir = os.path.join(maildir_path, "cur")
        can_mark_as_read = False
        if os.path.exists(cur_mail_dir):
             can_mark_as_read = os.access(new_mail_dir, os.W_OK) and os.access(cur_mail_dir, os.W_OK)
        
        # Mark the message as read by moving it from 'new' to 'cur'
        if most_recent_file is not None and most_recent_message is not None and can_mark_as_read:
            try:
                # Create a new filename with the :2,S suffix (S = Seen flag in Maildir)
                base_name = os.path.basename(most_recent_file)
                new_file_path = os.path.join(cur_mail_dir, base_name + ":2,S")
                
                # Move the file from 'new' to 'cur' with the seen flag
                os.rename(most_recent_file, new_file_path)
                logger.info(f"Marked email with subject '{most_recent_message['subject']}' as read")
            except Exception as e:
                logger.warning(f"Could not mark email as read: {e}. The email was still read successfully.")
        
        elif most_recent_message is not None and not can_mark_as_read:
            logger.warning(f"Email found but could not be marked as read: No write access to Maildir ({cur_mail_dir})")
        
        if not most_recent_message:
            logger.info("No relevant emails found after checking files.")

        return most_recent_message or {}
    
    except Exception as e:
        error_msg = f"Error checking mail: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def parse_email_commands(email_body: str) -> Dict[str, Any]:
    """Parse commands from an email body.
    
    Args:
        email_body: The body text of the email to parse
        
    Returns:
        Dictionary of commands and their values
    """
    commands = {}
    
    # Extract commands using regex patterns
    for command_key, pattern in COMMAND_PATTERNS.items():
        match = re.search(pattern, email_body, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            
            # Ignore if the value is the placeholder text for this command
            if command_key in COMMAND_TEMPLATE_PLACEHOLDERS and value == COMMAND_TEMPLATE_PLACEHOLDERS[command_key]:
                logger.debug(f"Ignoring placeholder value for command '{command_key}': {value}")
                continue
            
            # Ignore if value is empty after stripping (e.g., user deleted placeholder but left it blank)
            if not value:
                logger.debug(f"Ignoring empty value for command '{command_key}'.")
                continue

            # Convert to appropriate types
            if command_key == "frequency":
                try:
                    parsed_value = int(value)
                    if parsed_value <= 0:
                        logger.warning(f"Invalid frequency '{value}'. Must be a positive integer. Ignoring.")
                        continue
                    commands[command_key] = parsed_value
                except ValueError:
                    logger.warning(f"Could not parse frequency '{value}' as integer. Ignoring.")
                    continue # Skip adding this command if parsing fails
            elif command_key == "detail":
                if value.lower() in ["low", "medium", "high"]:
                    commands[command_key] = value.lower()
                else:
                    logger.warning(f"Invalid detail value '{value}'. Must be low, medium, or high. Ignoring.")
                    continue
            elif command_key in ["pause", "resume"]:
                if value.lower() == "true":
                    commands[command_key] = True
                elif value.lower() == "false":
                    commands[command_key] = False
                else:
                    logger.warning(f"Invalid boolean value '{value}' for command '{command_key}'. Must be true or false. Ignoring.")
                    continue
            else: # For 'focus' and 'feedback' which are strings
                commands[command_key] = value
    
    if commands:
        return commands
    else:
        return {"status": "No commands found in email"}

