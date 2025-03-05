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
from utils.agents.tools import get_timestamp
from utils.file_manager import FileManager
from smolagents import tool

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
    """Get the path for a paper file using the FileManager."""
    # Initialize file manager
    file_manager = FileManager()
    
    # Get the directory for paper files
    from utils.file_manager.file_manager import RESEARCH_PAPERS_DIR
    
    # Return the path in the research/papers directory
    return RESEARCH_PAPERS_DIR / f"{paper_id}{extension}"

def convert_pdf_to_markdown(paper_id: str, pdf_path: Path) -> None:
    """Convert PDF to Markdown in a separate thread."""
    try:
        logger.info(f"Starting conversion for {paper_id}")
        
        # Import here to avoid loading the library unless needed
        import pymupdf4llm
        
        markdown = pymupdf4llm.to_markdown(pdf_path, show_progress=False)
        md_path = get_paper_path(paper_id, ".md")
        
        # Initialize file manager
        file_manager = FileManager()
        
        # Save the markdown content using the file manager
        file_id = file_manager.save_file(
            content=markdown,
            file_type="paper",
            title=f"Paper_{paper_id}",
            metadata={"paper_id": paper_id, "source": "pdf_conversion"},
            extension=".md"
        )

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
def save_report(content: str, title: str = None) -> str:
    """Saves the research report to a file in the data/reports directory.
    
    This tool saves the provided content as a markdown file in the reports directory.
    The filename includes a timestamp and an optional title for easy identification.
    
    Args:
        content: The markdown content of the report to save
        title: Optional title to include in the filename
        
    Returns:
        A message confirming the report was saved and the path to the file
    """
    try:
        # Initialize file manager
        file_manager = FileManager()
        
        # Save the file using the file manager
        file_id = file_manager.save_file(
            content=content,
            file_type="report",
            title=title,
            metadata={"word_count": len(content.split())}
        )
        
        # Get the file metadata to return the path
        file_data = file_manager.get_file(file_id)
        report_path = file_data["metadata"]["filepath"]
        
        logger.info(f"Report saved to {report_path}")
        
        return f"Report successfully saved to {report_path}"
        
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return f"Error: Failed to save report: {str(e)}"

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
        
        # Save the PDF locally using the FileManager
        file_manager = FileManager()
        
        # Get the path where the PDF should be saved
        pdf_path = get_paper_path(paper_id)
        
        # Save the PDF content to the file
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
    try:
        # Initialize file manager
        file_manager = FileManager()
        
        # Try to find the file by searching for files with the paper_id in metadata
        paper_files = file_manager.list_files(
            file_type="paper", 
            filter_criteria={"paper_id": paper_id}
        )
        
        if paper_files:
            # Get the first matching file
            file_data = file_manager.get_file(paper_files[0]["file_id"])
            return file_data["content"]
        
        # If not found in metadata, try the traditional path approach as fallback
        md_path = get_paper_path(paper_id, ".md")
        
        if not md_path.exists():
            return f"Error: No markdown file found for paper ID {paper_id}"
        
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return content
    
    except Exception as e:
        logger.error(f"Error reading paper markdown: {str(e)}")
        return f"Error: Failed to read paper markdown: {str(e)}"

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