"""Example demonstrating how to create agents using the BaseAgent classes.

This example shows how to:
1. Create a simple ToolCalling agent
2. Create a CodeAgent with tools
3. Use the agents with automatic rate limiting and retry logic
"""

import sys
import os
from typing import List, Optional

# Add the project root to the path so we can import from agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agents.utils.agents.base_agent import BaseCodeAgent, BaseToolCallingAgent
from agents.utils.agents.tools import web_search, visit_webpage


class SimpleAnalystAgent(BaseToolCallingAgent):
    """A simple analyst agent that provides analysis without external tools."""
    
    def get_tools(self) -> List:
        """Return empty list - this agent doesn't use tools."""
        return []
    
    def get_base_description(self) -> str:
        """Return the base description for the analyst agent."""
        return "An analytical agent that provides insights and analysis on given topics."
    
    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the analyst agent."""
        return """You are an analytical agent that provides thorough analysis and insights on topics.

Your task is to:
1. Break down complex topics into key components
2. Identify important patterns and relationships
3. Provide clear, structured analysis
4. Draw meaningful conclusions

Provide your analysis in a clear, well-structured format with:
- Executive summary
- Key findings
- Detailed analysis
- Conclusions and recommendations

Be thorough but concise, and support your analysis with reasoning.
"""
    
    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving."""
        return "analyst"


class WebResearchAgent(BaseCodeAgent):
    """A web research agent that can search and analyze web content."""
    
    def get_tools(self) -> List:
        """Return the list of tools for web research."""
        return [web_search, visit_webpage]
    
    def get_base_description(self) -> str:
        """Return the base description for the web research agent."""
        return "A web research agent that can search the internet and analyze web content."
    
    def get_default_system_prompt(self) -> str:
        """Return the default system prompt for the web research agent."""
        return """You are a web research agent that can search the internet and analyze web content.

As a CodeAgent, you write Python code to call your tools:

1. Web Search:
```py
# Search for information
results = web_search("your search query", max_results=5)
print(results)
```

2. Visit Webpage:
```py
# Visit and extract content from a webpage
content = visit_webpage("https://example.com")
print(content)
```

Your task is to:
1. Perform targeted web searches based on the query
2. Visit relevant websites to gather detailed information
3. Analyze and synthesize the information
4. Provide a comprehensive report with sources

Always include source URLs in your final report.
"""
    
    def get_agent_type_name(self) -> str:
        """Return the agent type name for file saving."""
        return "web_research"


def main():
    """Demonstrate usage of the base agent classes."""
    print("=== Base Agent Example ===\n")
    
    # Example 1: Simple analyst agent (no tools)
    print("1. Creating SimpleAnalystAgent...")
    analyst = SimpleAnalystAgent(
        max_steps=5,
        agent_description="Specialized in market analysis and trends."
    )
    
    query1 = "Analyze the current trends in artificial intelligence adoption in healthcare."
    print(f"Query: {query1}")
    print("Running analyst agent...")
    
    try:
        result1 = analyst.run(query1)
        print(f"Result length: {len(result1)} characters")
        print(f"First 200 chars: {result1[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Web research agent (with tools)
    print("2. Creating WebResearchAgent...")
    researcher = WebResearchAgent(
        max_steps=10,
        agent_description="Specialized in gathering current web information.",
        additional_authorized_imports=["json", "re"]
    )
    
    query2 = "Research the latest developments in quantum computing from reliable sources."
    print(f"Query: {query2}")
    print("Running web research agent...")
    
    try:
        result2 = researcher.run(query2)
        print(f"Result length: {len(result2)} characters")
        print(f"First 200 chars: {result2[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Sharing a model between agents
    print("3. Sharing model between agents...")
    from agents.utils.agents.rate_lim_llm import RateLimitedLiteLLMModel
    
    shared_model = RateLimitedLiteLLMModel(
        model_id="gemini/gemini-2.0-flash",
        max_retries=2
    )
    
    analyst_shared = SimpleAnalystAgent(
        model=shared_model,
        max_steps=3
    )
    
    researcher_shared = WebResearchAgent(
        model=shared_model,
        max_steps=5
    )
    
    print("Both agents now share the same rate-limited model instance.")
    print("Rate limiting will be coordinated across both agents.")
    
    # Show rate limit status
    shared_model.print_rate_limit_status()


if __name__ == "__main__":
    main() 