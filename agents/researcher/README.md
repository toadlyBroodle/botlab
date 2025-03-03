# Researcher CodeAgent

The Researcher CodeAgent is designed to help with research tasks by providing tools to search the web, visit webpages, search arXiv, and process PDF documents. Unlike a regular agent, the CodeAgent writes Python code to call its tools, allowing for more complex and flexible research workflows.

## Tools

### Web Search
The `web_search` tool performs a DuckDuckGo web search with intelligent rate limiting to avoid being blocked.

### Visit Webpage
The `visit_webpage` tool visits a webpage at the given URL and returns its content as a markdown string.

### arXiv Search
The `arxiv_search` tool searches arXiv for research papers matching the given query.

### PDF to Markdown
The `pdf_to_markdown` tool downloads a PDF from a URL and converts it to markdown format. The conversion happens in a background thread to avoid blocking the agent.

### Save Report
The `save_report` tool saves the final research report to a file in the `data/reports` directory. The filename includes a timestamp and an optional title for easy identification.

## CodeAgent Usage

The CodeAgent writes Python code to call its tools. Here are examples of how it might use each tool:

```python
# Search for information on a topic
search_results = web_search("quantum computing applications", max_results=5)

# Visit a webpage and extract its content
webpage_content = visit_webpage("https://example.com/article")

# Search for papers on arXiv
arxiv_results = arxiv_search("transformer models", max_results=3, sort_by="relevance")

# Extract PDF URLs from arXiv results
pdf_urls = []
for line in arxiv_results.split('\n'):
    if line.startswith("**PDF:**"):
        pdf_url = line.split("**PDF:**")[1].strip()
        pdf_urls.append(pdf_url)

# Download and convert a PDF to markdown
pdf_result = pdf_to_markdown("https://arxiv.org/pdf/1706.03762.pdf")

# Extract the paper ID from the result
paper_id = None
for line in pdf_result.split('\n'):
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
        
        # Extract the status from the result
        for line in status_result.split('\n'):
            if line.startswith("Status:"):
                status = line.split(':')[1].strip()
                break
        
        if status == "processing":
            time.sleep(2)
    
    # Read the markdown content if conversion was successful
    if status == "success":
        markdown_content = read_paper_markdown(paper_id)

# Save the final research report to a file
final_report = """# Research Report: Quantum Computing
## Introduction
Quantum computing is a rapidly evolving field that leverages quantum mechanics...

## Sources
- https://example.com/quantum-computing
- https://arxiv.org/abs/2101.12345
"""
save_result = save_report(final_report, title="Quantum Computing Research")
```

## Running from the Command Line

You can run the researcher agent directly from the command line using the `main.py` script:

```bash
# Run with default query
python -m agents.researcher.main

# Run with a custom query
python -m agents.researcher.main --query "What are the latest advancements in quantum computing?"

# Run with custom parameters
python -m agents.researcher.main --max-steps 30 --base-wait-time 3.0 --max-retries 5 --model-id "gemini/gemini-2.0-pro"
```

### Command-line Arguments

- `--query`: The query to research (default: "What are the latest advancements in large language models? Include information from recent arXiv papers.")
- `--enable-telemetry`: Enable telemetry (flag)
- `--max-steps`: Maximum number of steps (default: 20)
- `--base-wait-time`: Base wait time for rate limiting (default: 2.0)
- `--max-retries`: Maximum retries for rate limiting (default: 3)
- `--model-id`: Model ID to use (default: "gemini/gemini-2.0-flash")
- `--model-info-path`: Path to model info JSON file (default: "utils/gemini/gem_llm_info.json")

## Using the Agent Programmatically

You can use the researcher CodeAgent programmatically in your own Python code in several ways:

### Option 1: Using the main function

```python
from agents.researcher.main import main

# Run the agent with command-line arguments
result = main()
```

### Option 2: Using initialize and run_agent_with_query

```python
from agents.researcher.main import initialize, run_agent_with_query

# Initialize the agent with custom parameters
researcher_agent = initialize(
    enable_telemetry=False,
    max_steps=30,
    base_wait_time=3.0,
    max_retries=5,
    model_id="gemini/gemini-2.0-flash",
    model_info_path="utils/gemini/gem_llm_info.json"
)

# Run a query with the agent
result = run_agent_with_query(
    researcher_agent, 
    "What are the latest advancements in quantum computing?",
)
```

### Option 3: Using the agent directly

```python
from agents.researcher.main import initialize

# Initialize the agent
researcher_agent = initialize()

# Run a query directly on the agent
result = researcher_agent.run("What are the latest advancements in quantum computing?")
```

## Example Script

The `example.py` script demonstrates different ways to use the researcher CodeAgent:

```python
from agents.researcher.main import initialize, run_agent_with_query, main as run_main

# Option 1: Use the main function from main.py
result = run_main()

# Option 2: Initialize the agent and run a custom query
agent = initialize(max_steps=20, model_id="gemini/gemini-2.0-flash")
custom_query = "What is quantum computing and how does it differ from classical computing?"
run_agent_with_query(agent, custom_query)
```

You can also run the example script with a custom query directly from the command line:

```bash
# Run with the default query
./agents/researcher/example.py

# Run with a custom query
./agents/researcher/example.py "What are the latest advancements in quantum computing?"

# Run with custom parameters
./agents/researcher/example.py "What are the latest advancements in quantum computing?" --max-steps 30 --model "gemini/gemini-2.0-pro"

# Skip running the main.py example
./agents/researcher/example.py --skip-main "What are the latest advancements in quantum computing?"
```

## Integration with Manager Agent

The researcher CodeAgent can be used as part of the Manager Agent system. The Manager Agent can coordinate multiple agents, including the researcher CodeAgent.

To use the researcher CodeAgent with the Manager Agent:

```python
from agents.manager.main import initialize as initialize_manager

# Initialize the manager agent with the researcher CodeAgent
run_query = initialize_manager(
    create_researcher=True,  # This will create and add the researcher CodeAgent
    max_steps=20,
    base_wait_time=2.0,
    max_retries=3,
    model_id="gemini/gemini-2.0-flash"
)

# Run a query through the manager agent
result = run_query("What are the latest advancements in quantum computing?")
```

You can also run the manager agent example script:

```bash
# Run with default query
python -m agents.manager.example

# Run with a custom query
python -m agents.manager.example "What are the latest advancements in quantum computing?"

# Run with advanced setup (custom configured researcher agent)
python -m agents.manager.example --advanced "What are the latest advancements in quantum computing?"
```

## Research Workflow

The researcher CodeAgent follows this workflow:

1. Start with an arXiv search for academic papers on the topic
2. Download and convert relevant PDFs to markdown for analysis
3. Follow up with web searches for additional context
4. Visit relevant webpages to extract detailed information
5. Compile findings into a comprehensive report
6. Save the final report to the `data/reports` directory using the `save_report` tool

The saved reports include a timestamp and optional title in the filename for easy identification.

## PDF to Markdown Workflow

The PDF to markdown functionality follows this workflow:

1. The agent writes code to call `pdf_to_markdown(url)` with a URL to a PDF document
2. The tool downloads the PDF and saves it locally with a unique ID
3. A background thread is started to convert the PDF to markdown using `pymupdf4llm`
4. The agent writes code to check the status of the conversion with `check_conversion_status(paper_id)`
5. Once the conversion is complete, the agent writes code to read the markdown with `read_paper_markdown(paper_id)`

The converted markdown files are stored in the `data/papers` directory.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Or if you're using Poetry:

```bash
poetry install
```

## Dependencies

- pymupdf4llm: For converting PDFs to markdown
- requests: For downloading files and making HTTP requests
- arxiv: For searching and retrieving papers from arXiv
- smolagents: For the base agent functionality 