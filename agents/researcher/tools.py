import time
import random
import arxiv
import os
import requests
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from smolagents import tool
from smolagents import DuckDuckGoSearchTool
from smolagents import VisitWebpageTool

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Directory for storing papers
PAPERS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

# Dictionary to track conversion statuses
conversion_statuses = {}

class ConversionStatus:
    """Class to track the status of PDF to markdown conversions."""
    def __init__(self, paper_id: str, url: str):
        self.paper_id = paper_id
        self.url = url
        self.status = "pending"  # pending, processing, success, error
        self.started_at = datetime.now()
        self.completed_at = None
        self.error = None

def get_paper_path(paper_id: str, extension: str = ".pdf") -> Path:
    """Get the path for a paper file."""
    return PAPERS_DIR / f"{paper_id}{extension}"

def convert_pdf_to_markdown(paper_id: str, pdf_path: Path) -> None:
    """Convert PDF to Markdown in a separate thread."""
    try:
        logger.info(f"Starting conversion for {paper_id}")
        
        # Import here to avoid loading the library unless needed
        import pymupdf4llm
        
        markdown = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        md_path = get_paper_path(paper_id, ".md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        status = conversion_statuses.get(paper_id)
        if status:
            status.status = "success"
            status.completed_at = datetime.now()
            
        # Clean up PDF after successful conversion
        pdf_path.unlink()
        logger.info(f"Conversion completed for {paper_id}")
        
    except Exception as e:
        logger.error(f"Conversion failed for {paper_id}: {str(e)}")
        status = conversion_statuses.get(paper_id)
        if status:
            status.status = "error"
            status.completed_at = datetime.now()
            status.error = str(e)

@tool
def pdf_to_markdown(url: str) -> str:
    """Downloads a PDF from a URL and converts it to markdown format.
    
    This tool fetches a PDF document from the provided URL, saves it locally,
    and then converts it to markdown format using pymupdf4llm. The conversion
    happens in a background thread to avoid blocking the agent.
    
    Args:
        url: The URL of the PDF document to download and convert
        
    Returns:
        A message with the status of the operation and the ID of the paper
        that can be used to check the conversion status or retrieve the markdown
    """
    try:
        # Generate a unique ID for this paper
        paper_id = str(uuid.uuid4())[:8]
        
        # Create a status object to track the conversion
        status = ConversionStatus(paper_id, url)
        conversion_statuses[paper_id] = status
        
        # Download the PDF
        logger.info(f"Downloading PDF from {url}")
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            status.status = "error"
            status.completed_at = datetime.now()
            status.error = f"Failed to download PDF: HTTP {response.status_code}"
            return f"Error: Failed to download PDF from {url}. HTTP status code: {response.status_code}"
        
        # Check if the content type is PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            status.status = "error"
            status.completed_at = datetime.now()
            status.error = f"URL does not point to a PDF file: {content_type}"
            return f"Error: URL does not appear to point to a PDF file. Content-Type: {content_type}"
        
        # Save the PDF locally
        pdf_path = get_paper_path(paper_id)
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Update status
        status.status = "processing"
        
        # Start conversion in a separate thread
        thread = threading.Thread(
            target=convert_pdf_to_markdown,
            args=(paper_id, pdf_path)
        )
        thread.daemon = True
        thread.start()
        
        return (
            f"PDF download successful. Converting to markdown in the background.\n"
            f"Paper ID: {paper_id}\n"
            f"You can check the status with check_conversion_status('{paper_id}')\n"
            f"Once complete, you can read the markdown with read_paper_markdown('{paper_id}')"
        )
        
    except Exception as e:
        logger.error(f"Error in pdf_to_markdown: {str(e)}")
        return f"Error: Failed to process PDF: {str(e)}"

@tool
def check_conversion_status(paper_id: str) -> str:
    """Check the status of a PDF to markdown conversion.
    
    Args:
        paper_id: The ID of the paper conversion to check
        
    Returns:
        A message with the current status of the conversion
    """
    status = conversion_statuses.get(paper_id)
    if not status:
        return f"Error: No conversion found with ID {paper_id}"
    
    result = f"Conversion status for {paper_id}:\n"
    result += f"Status: {status.status}\n"
    result += f"Started at: {status.started_at}\n"
    
    if status.completed_at:
        result += f"Completed at: {status.completed_at}\n"
        
    if status.error:
        result += f"Error: {status.error}\n"
        
    return result

@tool
def read_paper_markdown(paper_id: str) -> str:
    """Read the markdown content of a converted paper.
    
    Args:
        paper_id: The ID of the paper to read
        
    Returns:
        The markdown content of the paper, or an error message if not available
    """
    md_path = get_paper_path(paper_id, ".md")
    
    if not md_path.exists():
        return f"Error: No markdown file found for paper ID {paper_id}"
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return content

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

# Global variables to track search state for rate limiting
_last_search_time = 0
_consecutive_failures = 0
_base_wait_time = 2.0

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
        print(f"Waiting {sleep_time:.2f} seconds to respect DDGS rate limits")
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
                    
                    print(f"Empty result or error detected. Retrying in {backoff_time:.2f} seconds...")
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
                    
                    print(f"Rate limit detected. Retrying in {backoff_time:.2f} seconds...")
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
def arxiv_search(
    query: str, 
    max_results: int = 10, 
    sort_by: str = "relevance", 
    sort_order: str = "descending",
    full_text: bool = False,
    include_categories: Optional[List[str]] = None
) -> str:
    """Searches arXiv for research papers matching the given query.
    
    This tool provides access to arXiv's database of research papers, preprints, and scientific 
    articles across various fields including Computer Science, Physics, Mathematics, and more.
    
    Query Tips:
    - Use quotes for exact phrases: "quantum computing"
    - Use AND/OR/NOT for boolean logic: quantum AND computing
    - Use parentheses for grouping: (quantum OR neural) AND network
    - Use categories with prefix: cat:cs.AI (Artificial Intelligence), cat:physics.gen-ph (General Physics)
    - Use author search: au:lastname (author lastname search)
    
    Common arXiv categories:
    - cs.AI: Artificial Intelligence
    - cs.CL: Computation and Language (NLP)
    - cs.CV: Computer Vision
    - cs.LG: Machine Learning
    - cs.NE: Neural and Evolutionary Computing
    - quant-ph: Quantum Physics
    - math.ST: Statistics Theory
    - physics.data-an: Data Analysis, Statistics and Probability
    - stat.ML: Machine Learning Statistics
    
    Args:
        query: The search query string
        max_results: Maximum number of papers to return (default: 10)
        sort_by: Sorting method - "relevance", "lastUpdatedDate", or "submittedDate" (default: "relevance")
        sort_order: Sorting order - "ascending" or "descending" (default: "descending")
        full_text: Whether to search the full text of papers, not just metadata (default: False)
        include_categories: Optional list of arXiv categories to limit the search to (e.g. ["cs.AI", "cs.CL"])
        
    Returns:
        Formatted search results as a markdown string with titles, authors, abstracts, and URLs
    """
    # Configure the search parameters
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion[sort_by.upper()] if sort_by.upper() in dir(arxiv.SortCriterion) else arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder[sort_order.upper()] if sort_order.upper() in dir(arxiv.SortOrder) else arxiv.SortOrder.Descending,
    )
    
    # Execute the search and format results
    results = []
    try:
        client = arxiv.Client()
        papers = list(client.results(search))
        
        if not papers:
            return "No results found for your query. Try broadening your search or using different keywords."
        
        results.append(f"# arXiv Search Results for: {query}\n")
        results.append(f"Found {len(papers)} papers. Showing top {min(max_results, len(papers))}:\n")
        
        for i, paper in enumerate(papers[:max_results], 1):
            # Format categories
            categories = ", ".join(paper.categories)
            
            # Format published and updated dates
            published = paper.published.strftime("%Y-%m-%d") if paper.published else "N/A"
            updated = paper.updated.strftime("%Y-%m-%d") if paper.updated else "N/A"
            
            # Format authors (limit to first 5 if many)
            authors = paper.authors
            if len(authors) > 5:
                author_text = ", ".join(str(author) for author in authors[:5]) + f" and {len(authors)-5} more"
            else:
                author_text = ", ".join(str(author) for author in authors)
            
            # Create paper entry
            results.append(f"## {i}. {paper.title}")
            results.append(f"**Authors:** {author_text}")
            results.append(f"**Published:** {published} | **Last Updated:** {updated}")
            results.append(f"**Categories:** {categories}")
            results.append(f"**arXiv ID:** {paper.entry_id.split('/')[-1]}")
            results.append(f"**URL:** {paper.entry_id}")
            results.append(f"**PDF:** {paper.pdf_url}")
            results.append("\n**Abstract:**")
            results.append(paper.summary.replace("\n", " ").strip())
            results.append("\n---\n")
        
        return "\n".join(results)
    
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"