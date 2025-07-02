# Base Agent System Guide

## Overview

The Base Agent system provides a unified foundation for all specialized agents in the botlab project. It encapsulates common functionality including rate limiting, retry logic, model management, and execution patterns, allowing individual agents to focus only on their specific business logic.

## Key Benefits

### ✅ **Before vs After**

**Before (Old Agent Pattern):**
- 200+ lines of boilerplate code per agent
- Duplicate rate limiting logic
- Inconsistent error handling
- Manual model initialization
- Repetitive retry mechanisms
- Scattered execution timing and result saving

**After (With BaseAgent):**
- 50-80 lines per agent (60-75% reduction)
- Shared, tested rate limiting
- Consistent error handling across all agents
- Automatic model management
- Unified retry logic with exponential backoff
- Standardized execution framework

### 🔄 **Code Reduction Example**

The `ResearcherAgent` went from **277 lines** to **145 lines** (48% reduction) by using `BaseCodeAgent`.

## Architecture

```
BaseAgent (Abstract)
├── BaseCodeAgent (for CodeAgent-based agents)
└── BaseToolCallingAgent (for ToolCallingAgent-based agents)
```

## Core Features

### 1. **Automatic Rate Limiting**
- Shared `RateLimitedLiteLLMModel` with global rate limit tracking
- Model fallback support (e.g., Gemini Pro → Flash)
- Dynamic minimum delays between calls
- Cooldown periods for models hitting limits

### 2. **Intelligent Retry Logic**
- Exponential backoff with jitter
- API-specified retry delays (parses `retryDelay` from errors)
- Distinguishes between rate limit and other errors
- Configurable max retries and base wait times

### 3. **Execution Framework**
- Automatic execution timing
- Standardized result saving with `save_final_answer`
- Consistent logging across all agents
- Error handling and reporting

### 4. **Flexible Configuration**
- Model sharing between agents
- Custom system prompts and descriptions
- Additional tools and imports
- Managed agents support

## Creating a New Agent

### Simple ToolCalling Agent

```python
from agents.utils.agents.base_agent import BaseToolCallingAgent
from typing import List

class MyAnalystAgent(BaseToolCallingAgent):
    """An agent that provides analysis without external tools."""
    
    def get_tools(self) -> List:
        return []  # No tools needed
    
    def get_base_description(self) -> str:
        return "An analytical agent that provides insights on topics."
    
    def get_default_system_prompt(self) -> str:
        return """You are an analytical agent that provides analysis.
        
        Your task is to:
        1. Analyze the given topic
        2. Provide structured insights
        3. Draw meaningful conclusions
        """
    
    def get_agent_type_name(self) -> str:
        return "analyst"  # Used for file saving

# Usage
agent = MyAnalystAgent(max_steps=5)
result = agent.run("Analyze AI trends in healthcare")
```

### CodeAgent with Tools

```python
from agents.utils.agents.base_agent import BaseCodeAgent
from agents.utils.agents.tools import web_search, visit_webpage
from typing import List

class WebResearchAgent(BaseCodeAgent):
    """An agent that researches topics using web search."""
    
    def get_tools(self) -> List:
        return [web_search, visit_webpage]
    
    def get_base_description(self) -> str:
        return "A web research agent that searches and analyzes web content."
    
    def get_default_system_prompt(self) -> str:
        return """You are a web research agent.
        
        Use your tools:
        ```py
        results = web_search("your query", max_results=5)
        content = visit_webpage("https://example.com")
        ```
        
        Always provide sources in your final report.
        """
    
    def get_agent_type_name(self) -> str:
        return "web_research"

# Usage with custom configuration
agent = WebResearchAgent(
    max_steps=10,
    agent_description="Specialized in tech research.",
    additional_authorized_imports=["json", "re"]
)
result = agent.run("Research quantum computing developments")
```

## Migrating Existing Agents

### Step 1: Choose Base Class
- Use `BaseCodeAgent` if your agent uses `CodeAgent`
- Use `BaseToolCallingAgent` if your agent uses `ToolCallingAgent`

### Step 2: Extract Required Methods
```python
# Old pattern
class OldAgent:
    def __init__(self, ...):
        # 50+ lines of boilerplate
        
# New pattern  
class NewAgent(BaseCodeAgent):
    def __init__(self, ...):
        super().__init__(...)  # 1 line replaces 50+
    
    def get_tools(self) -> List:
        return [tool1, tool2]  # Just return your tools
    
    def get_base_description(self) -> str:
        return "Your agent description"
    
    def get_default_system_prompt(self) -> str:
        return "Your system prompt"
```

### Step 3: Remove Boilerplate
Delete all the common code:
- Model initialization
- Agent creation
- Custom prompt application
- Retry logic
- Execution timing
- Result saving

### Step 4: Update Method Calls
```python
# Old
def run_query(self, query):
    # 50+ lines of retry logic, timing, saving
    
# New
def run_query(self, query):
    return self.run(query)  # Base class handles everything
```

## Advanced Usage

### Sharing Models Between Agents

```python
from agents.utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

# Create shared model
shared_model = RateLimitedLiteLLMModel(
    model_id="gemini/gemini-2.0-flash",
    max_retries=3
)

# Share across agents
researcher = ResearcherAgent(model=shared_model)
writer = WriterAgent(model=shared_model)
analyst = AnalystAgent(model=shared_model)

# All agents coordinate rate limiting
```

### Custom Execution Logic

```python
class CustomAgent(BaseCodeAgent):
    # ... implement required methods ...
    
    def execute_task(self, task: str) -> str:
        """Override for custom execution logic."""
        # Pre-processing
        processed_task = self.preprocess(task)
        
        # Use base agent
        result = self.agent.run(processed_task)
        
        # Post-processing
        return self.postprocess(result)
```

### Agent with Managed Sub-Agents

```python
class ManagerAgent(BaseCodeAgent):
    def __init__(self, sub_agents, **kwargs):
        super().__init__(
            managed_agents=[agent.agent for agent in sub_agents],
            **kwargs
        )
    
    def get_tools(self) -> List:
        return []  # Uses managed agents instead of tools
```

## Configuration Options

### Constructor Parameters

```python
agent = BaseAgent(
    model=None,                          # Optional shared model
    max_steps=20,                        # Agent step limit
    agent_description="Custom desc",     # Appends to base description
    system_prompt="Custom prompt",       # Overrides default prompt
    model_id="gemini/gemini-2.0-flash",  # Model if creating new
    model_info_path="path/to/limits.json", # Rate limit config
    base_wait_time=2.0,                  # Retry backoff base
    max_retries=3,                       # Max retry attempts
    additional_tools=[custom_tool],      # Extra tools
    additional_authorized_imports=["json"], # Extra imports for CodeAgent
    managed_agents=[sub_agent],          # Managed sub-agents
    agent_name="custom_name"             # Override default name
)
```

### Method Overrides

```python
class CustomAgent(BaseCodeAgent):
    def get_agent_type_name(self) -> str:
        """Override for custom file saving prefix."""
        return "custom_prefix"
    
    def execute_task(self, task: str) -> str:
        """Override for custom execution logic."""
        return super().execute_task(task)
```

## Rate Limiting Features

### Automatic Rate Limit Management
- **Shared Tracking**: All agent instances share rate limit counters
- **Safety Buffers**: Operates at 93% of API limits by default
- **Dynamic Delays**: Calculates minimum delays based on model RPM
- **Cooldown Periods**: Temporary bans after hitting limits

### Error Handling
- **Smart Retry Logic**: Distinguishes rate limit from other errors
- **API Delay Parsing**: Uses `retryDelay` from API error responses
- **Exponential Backoff**: Falls back to exponential backoff with jitter
- **Fallback Models**: Automatic switching to lower-tier models

### Monitoring
```python
# Check rate limit status
agent.model.print_rate_limit_status()

# Get detailed status
status = agent.model.shared_tracker.get_rate_limit_status(model_id)
```

## File Organization

```
agents/
├── utils/agents/
│   ├── base_agent.py       # Main base classes
│   ├── tools.py           # Shared agent tools
│   └── __init__.py        # Exports base classes
├── researcher/
│   └── agents.py          # ResearcherAgent (refactored)
├── writer_critic/
│   └── agents.py          # WriterAgent, CriticAgent (refactored)
└── base_agent_example.py  # Usage examples
```

## Best Practices

### 1. **Keep It Simple**
- Only implement the 4 required methods
- Use base class defaults when possible
- Don't override `execute_task()` unless necessary

### 2. **Share Models**
- Create one `RateLimitedLiteLLMModel` per application
- Pass it to all agents for coordinated rate limiting
- Monitor usage with `print_rate_limit_status()`

### 3. **Naming Conventions**
- Agent classes: `YourFeatureAgent`
- File prefixes: Return short, clear names from `get_agent_type_name()`
- Tool functions: Use descriptive names

### 4. **Error Handling**
- Let base class handle rate limits and retries
- Only catch agent-specific business logic errors
- Use logging for debugging

### 5. **Testing**
- Test your 4 required methods independently
- Use base class for integration testing
- Mock rate limiting in unit tests

## Migration Checklist

- [ ] Choose appropriate base class (`BaseCodeAgent` or `BaseToolCallingAgent`)
- [ ] Implement 4 required methods: `get_tools()`, `get_base_description()`, `get_default_system_prompt()`, `get_agent_type_name()`
- [ ] Update constructor to call `super().__init__(...)`
- [ ] Remove all boilerplate code (model init, agent creation, retry logic, etc.)
- [ ] Update method signatures to use base class `run()` method
- [ ] Test that functionality remains the same
- [ ] Update imports to include base classes
- [ ] Update documentation and examples

## Examples

See `base_agent_example.py` for complete working examples of:
- Simple ToolCalling agent
- CodeAgent with tools  
- Model sharing between agents
- Custom configuration options

## Support

The base agent system is designed to be backwards compatible. Existing agents can be migrated incrementally, and the old patterns will continue to work during the transition period. 