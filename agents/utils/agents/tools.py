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
_base_wait_time = 2.0

def apply_custom_agent_prompts(agent) -> None:
    """Load and apply custom agent templates based on the agent type.
    
    Args:
        agent: The agent instance
        
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
    
    # Reinitialize the system prompt to apply the new template
    agent.system_prompt = agent.initialize_system_prompt()

    # append todays date and time to the system prompt
    agent.system_prompt += f"\n\nToday's date and time is {datetime.now().strftime('%Y-%m-%d %H:%M')}."

@tool
def web_search(query: str, max_results: int = 10, rate_limit_seconds: float = 2.0, max_retries: int = 3) -> str:
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
        rate_limit_seconds: Minimum seconds to wait between searches (default: 2.0)
        max_retries: Maximum number of retry attempts when rate limited (default: 3)
        
    Returns:
        Formatted search results as a markdown string with titles, URLs, and snippets
    """
    global _last_search_time, _consecutive_failures, _base_wait_time

    if rate_limit_seconds < 2.0:
        rate_limit_seconds = 2.0
    
    # Calculate current wait time with exponential backoff
    current_wait_time = rate_limit_seconds * (2 ** _consecutive_failures)
    # Add jitter to avoid synchronized requests
    current_wait_time = current_wait_time * (0.8 + 0.4 * random.random())
    
    # Apply rate limiting
    current_time = time.time()
    time_since_last_search = current_time - _last_search_time
    
    if time_since_last_search < current_wait_time:
        # Wait the remaining time to respect the rate limit
        sleep_time = current_wait_time - time_since_last_search
        logger.info(f"Waiting {sleep_time:.2f} seconds to respect DDGS rate limits")
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
                    # Calculate backoff time
                    backoff_time = _base_wait_time * (2 ** _consecutive_failures)
                    # Add jitter (±20%)
                    backoff_time = backoff_time * (0.8 + 0.4 * random.random())
                    
                    logger.info(f"Empty result or error detected. Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                    continue
                else:
                    return f"Error: DuckDuckGo search failed after {max_retries} retries. Last result: {result}"
            
            # Success - reset consecutive failures and update last search time
            _consecutive_failures = 0
            _last_search_time = time.time()
            
            return result
            
        except Exception as e:
            error_message = str(e)
            _last_search_time = time.time()
            
            # Check if it's a rate limit error - look for various indicators
            if any(term in error_message.lower() for term in ["ratelimit", "rate limit", "429", "too many requests", "202"]):
                _consecutive_failures += 1
                retries += 1
                
                if retries <= max_retries:
                    # Calculate backoff time with increased base wait time
                    backoff_time = _base_wait_time * (2 ** _consecutive_failures)
                    # Add jitter (±20%)
                    backoff_time = backoff_time * (0.8 + 0.4 * random.random())
                    
                    logger.info(f"Rate limit detected. Retrying in {backoff_time:.2f} seconds...")
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
