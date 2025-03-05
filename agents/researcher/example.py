#!/usr/bin/env python3
import time
import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the main module
import researcher.main as researcher_main

def main():
    """Example script to demonstrate how to use the researcher CodeAgent.
    
    This script shows how to use the researcher agent with custom parameters and prompts.
    """
    
    # Define specific system prompt for the researcher agent
    researcher_prompt = """You are an advanced AI research assistant specialized in gathering comprehensive information on technical and scientific topics. Your primary goal is to provide detailed, accurate, and well-sourced information by leveraging web searches and academic papers.

When researching technical or scientific topics, follow this workflow:
1. Start with an arXiv search using proper search syntax to find academic papers
2. For particularly relevant papers, download and convert them to markdown
3. Follow up with web searches for additional context, explanations, or recent developments
4. Visit relevant webpages to extract detailed information
5. Compile findings into a comprehensive report with clear sections
6. Always include all source URLs for all information

Your research reports should be well-structured with:
- An executive summary/introduction
- Key findings organized by subtopic
- Technical details with appropriate depth
- Recent developments and future directions
- A comprehensive list of sources

For any query, aim to find at least 5-10 high-quality, authoritative sources before compiling your report.
Always save your completed comprehensive report using the save_report tool before providing your final answer."""

    # Define specific description for the researcher agent
    researcher_description = "An advanced AI research assistant specialized in gathering comprehensive information on technical and scientific topics from both web searches and academic papers."

    # Initialize researcher system with custom prompts and descriptions
    run_research_query = researcher_main.initialize(
        max_steps=20, 
        enable_telemetry=False,
        researcher_description=researcher_description,
        researcher_prompt=researcher_prompt
    )
    
    # Example research query
    query = """What are the latest advancements in quantum computing? 
    Focus on breakthroughs in the last 2 years, including hardware developments, 
    error correction techniques, and potential applications. Include information 
    from recent arXiv papers and reputable sources."""
    
    start_time = time.time()
    
    # Run the research query
    result = run_research_query(query)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # Print path to saved report
    print("\nThe research report was saved in the shared_data/reports directory.")

if __name__ == "__main__":
    main() 