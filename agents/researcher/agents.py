from smolagents import CodeAgent
from smolagents.utils import AgentGenerationError
from .tools import arxiv_search, pdf_to_markdown, check_conversion_status, read_paper_markdown
from ..utils.agents.tools import web_search, visit_webpage, apply_custom_agent_prompts, save_final_answer
from ..utils.agents.base_agent import BaseCodeAgent
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel, parse_retry_delay_from_error
from ..utils.gemini.simple_llm import SimpleLiteLLMModel
from typing import Optional, List, Union
import os
import time
import random
import logging

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseCodeAgent):
    """A researcher agent that can craft advanced search queries and perform web searches using DuckDuckGo and arXiv."""
    
    def __init__(
        self,
        model: Optional[Union[RateLimitedLiteLLMModel, SimpleLiteLLMModel]] = None,
        max_steps: int = 20,
        researcher_description: Optional[str] = None,
        researcher_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        additional_tools: Optional[List] = None,
        disable_default_web_search: bool = False,  # New parameter to disable default web_search
        web_search_disabled: bool = False,  # New parameter to completely disable web search
        **kwargs  # Accept additional arguments to pass to BaseCodeAgent
    ):
        """Initialize the ResearcherAgent.
        
        Args:
            model: Optional LiteLLM model to use
            max_steps: Maximum number of steps for the agent
            researcher_description: Optional additional description
            researcher_prompt: Optional custom system prompt
            model_id: Model ID if creating a new model
            model_info_path: Path to model info JSON file
            base_wait_time: Base wait time for rate limiting (only used with rate limiting)
            max_retries: Maximum retries for rate limiting (only used with rate limiting)
            additional_tools: Optional list of additional tools
            disable_default_web_search: If True, excludes the default web_search tool from base tools
            web_search_disabled: If True, replaces web_search with a disabled placeholder
            **kwargs: Additional arguments to pass to BaseCodeAgent
        """
        # Store configuration for tool selection
        self.disable_default_web_search = disable_default_web_search
        self.web_search_disabled = web_search_disabled
        
        super().__init__(
            model=model,
            max_steps=max_steps,
            agent_description=researcher_description,
            system_prompt=researcher_prompt,
            model_id=model_id,
            model_info_path=model_info_path,
            base_wait_time=base_wait_time,
            max_retries=max_retries,
            additional_tools=additional_tools,
            additional_authorized_imports=["time", "json", "re", "uuid"],
            agent_name='researcher_agent',
            **kwargs  # Pass through additional arguments
        )

    def get_tools(self) -> List:
        """Return the list of tools for the researcher agent."""
        from ..utils.agents.tools import web_search, visit_webpage
        from smolagents import tool
        
        # Import arxiv tools with fallback if not available
        try:
            from ..utils.agents.tools import arxiv_search, pdf_to_markdown, check_conversion_status, read_paper_markdown
            arxiv_tools = [arxiv_search, pdf_to_markdown, check_conversion_status, read_paper_markdown]
        except ImportError:
            # Arxiv tools not available, use empty list
            arxiv_tools = []
        
        base_tools = [
            visit_webpage,
        ] + arxiv_tools
        
        # Conditionally add web_search based on configuration
        if self.web_search_disabled:
            # Create a disabled web_search placeholder locally to avoid circular imports
            @tool
            def web_search_disabled(query: str, max_results: int = 10, rate_limit_seconds: float = 5.0, max_retries: int = 3) -> str:
                """Placeholder web_search tool that returns an error when web search is completely disabled.
                
                Args:
                    query: The search query that would be performed (ignored since search is disabled)
                    max_results: Maximum number of results that would be returned (default: 10, ignored)
                    rate_limit_seconds: Minimum seconds that would wait between searches (default: 5.0, ignored)
                    max_retries: Maximum number of retry attempts when rate limited (default: 3, ignored)
                    
                Returns:
                    Error message indicating that web search is disabled for this agent
                """
                return "Error: Web search is disabled for this agent. Cannot perform search queries."
            
            # Set the function name to web_search to override the default tool
            web_search_disabled.__name__ = "web_search"
            base_tools.insert(0, web_search_disabled)
        elif not self.disable_default_web_search:
            # Add the default web_search tool (normal behavior)
            base_tools.insert(0, web_search)
        # If disable_default_web_search is True, we don't add any web_search here
        # The additional_tools should provide the replacement (web_search_gemini_only)
        
        return base_tools

    def get_base_description(self) -> str:
        """Return the base description for the researcher agent."""
        return """This agent can craft advanced search queries, perform web searches using DuckDuckGo and arXiv, and scrape resulting urls (including pdfs) content into markdown. Use this agent to research topics, find specific information, analyze specific webpage content, search for academic papers on arXiv, and process PDF documents."""

    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the researcher agent."""
        return """You are a researcher agent that can craft advanced search queries and perform web searches using DuckDuckGo and search for academic papers on arXiv. 
            You follow up all relevant search results by calling your `visit_webpage` tool and extracting the relevant content into a detailed markdown report, including all possibly relevant information.

As a CodeAgent, you write Python code to call your tools. Here are examples of how to call each tool:

1. Web Search:
```py
# Search for information on a topic
search_results = web_search("quantum computing applications", max_results=5)
print(search_results)
```

2. Visit Webpage:
```py
# Visit a webpage and extract its content
webpage_content = visit_webpage("https://example.com/article")
print(webpage_content)
```

3. arXiv Search:
```py
# Search for papers on arXiv
arxiv_results = arxiv_search("transformer models", max_results=3, sort_by="relevance")
print(arxiv_results)

# Extract PDF URLs from arXiv results
pdf_urls = []
for line in arxiv_results.split('\\n'):
    if line.startswith("**PDF:**"):
        pdf_url = line.split("**PDF:**")[1].strip()
        pdf_urls.append(pdf_url)
```

4. PDF to Markdown:
```py
# Download and convert a PDF to markdown
pdf_result = pdf_to_markdown("https://arxiv.org/pdf/1706.03762.pdf")
print(pdf_result)

# Extract the paper ID from the result
paper_id = None
for line in pdf_result.split('\\n'):
    if line.startswith("Paper ID:"):
        paper_id = line.split(':')[1].strip()
        break

# Check the conversion status
if paper_id:
    # Wait for the conversion to complete
    import time
    status = "processing"
    while status == "processing":
        status_result = check_conversion_status(paper_id)
        print(status_result)
        
        # Extract the status from the result
        for line in status_result.split('\\n'):
            if line.startswith("Status:"):
                status = line.split(':')[1].strip()
                break
        
        if status == "processing":
            print("Conversion still in progress. Waiting 2 seconds...")
            time.sleep(2)
    
    # Read the markdown content if conversion was successful
    if status == "success":
        markdown_content = read_paper_markdown(paper_id)
        print(markdown_content[:500] + "...") # Print first 500 characters
```

For academic, scientific, or research-oriented queries, you should first use the `arxiv_search` tool to find relevant papers and research. 
The arXiv search supports advanced query syntax like boolean operators (AND, OR, NOT), exact phrase matching with quotes, and category filtering (e.g., cat:cs.AI for AI papers).

When researching technical or scientific topics, follow this workflow:
1. Start with an arXiv search using proper search syntax
2. For particularly relevant papers, download and convert them to markdown using `pdf_to_markdown`
3. Check the conversion status with `check_conversion_status` and read the content with `read_paper_markdown`
4. Follow up with web searches for additional context or explanations
5. Visit relevant webpages to extract detailed information
6. Compile findings into a comprehensive report

If there isn't enough relevant information returned from a search, continue running improved searches (more specific, using advanced search operators) until you have enough information (aim for 10 high quality, authoritative sources). Do NOT focus on safety and ethical issues (these will be addressed elsewhere); only focus on the technical and factual details needed to answer the original query.

NEVER directly plagiarize content from sources, but instead use them to inform your own work and synthesize the information into your own unique report.
ALWAYS include ALL relevant source URLs for ALL information you use in your response, as this is the entire point of your task!

IMPORTANT: Your final_answer MUST include your complete, comprehensive research report in markdown format with the following sections:
1. A clear title related to the research query
2. An introduction explaining the topic and scope
3. Main content with findings organized into logical sections
4. A conclusion summarizing key points
5. A complete list of sources INCLUDING URLs!

The report should be well-structured and ready to be used as a standalone document. Your final_answer will be automatically saved as a report file, so make sure it contains your complete research findings.
"""

    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving."""
        return "researcher"
    
    def run_query(self, query: str, max_retries: Optional[int] = None, base_wait_time: Optional[float] = None) -> str:
        """Run the agent with a research query and return the result.
        
        This is a convenience method that calls the base run() method with the same signature
        as the original ResearcherAgent for backward compatibility.
        
        Args:
            query: The research query to run
            max_retries: Maximum number of retries for AgentGenerationError
            base_wait_time: Base wait time for exponential backoff if no retryDelay found
            
        Returns:
            The result from the agent containing the research report
        """
        return self.run(query, max_retries, base_wait_time)

