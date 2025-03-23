from smolagents import CodeAgent
from .tools import arxiv_search, pdf_to_markdown, check_conversion_status, read_paper_markdown
from agents.utils.agents.tools import web_search, visit_webpage, apply_custom_agent_prompts, save_final_answer
from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from typing import Optional
import os
import time


class ResearcherAgent:
    """A wrapper class for the researcher agent that handles initialization and query execution."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 20,
        researcher_description: Optional[str] = None,
        researcher_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the researcher agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the agent
            researcher_description: Optional additional description to append to the base description
            researcher_prompt: Optional custom system prompt to use instead of the default
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
        """
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        self.max_steps = max_steps
        
        # Create the agent
        base_description = """This agent can craft advanced search queries, perform web searches using DuckDuckGo and arXiv, and scrape resulting urls (including pdfs) content into markdown. Use this agent to research topics, find specific information, analyze specific webpage content, search for academic papers on arXiv, and process PDF documents."""
        
        # Append additional description if provided
        if researcher_description:
            description = f"{base_description} {researcher_description}"
        else:
            description = base_description

        tools = [
            web_search,
            visit_webpage,
            arxiv_search,
            pdf_to_markdown,
            check_conversion_status,
            read_paper_markdown
        ]
        
        self.agent = CodeAgent(
            tools=tools,
            model=self.model,
            additional_authorized_imports=["time", "json", "re", "uuid"],
            name='researcher_agent',
            description=description,
            max_steps=self.max_steps
        )

        # Default system prompt if none provided
        default_system_prompt = """You are a researcher agent that can craft advanced search queries and perform web searches using DuckDuckGo and search for academic papers on arXiv. 
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

        # Apply custom templates with the appropriate system prompt
        custom_prompt = researcher_prompt if researcher_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
    
    def run_query(self, query: str) -> str:
        """Run the agent with a research query and return the result.
        
        Args:
            query: The research query to run
            
        Returns:
            The result from the agent containing the research report
        """
        # Time the query execution
        start_time = time.time()
        
        # Run the query directly on the agent
        result = self.agent.run(query)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the final answer using the shared tool
        save_final_answer(
            agent=self.agent,
            result=result,
            query_or_prompt=query,
            agent_type="researcher"
        )
        
        return result

