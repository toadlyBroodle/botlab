from typing import List, Optional, Dict, Any, Tuple
from smolagents import CodeAgent
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from utils.agents.tools import apply_custom_agent_prompts
from .tools import select_best_output

def create_qaqc_agent(
    model: RateLimitedLiteLLMModel,
    max_steps: int = 15,
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> CodeAgent:
    """Creates a QAQC agent that compares outputs and selects the best one
    
    Args:
        model: The RateLimitedLiteLLMModel model to use for the agent
        max_steps: Maximum steps the agent can take
        agent_description: Optional custom description for the agent
        system_prompt: Optional custom system prompt for the agent
        
    Returns:
        A configured QAQC agent
    """
    
    # Use custom description if provided, otherwise use default
    description = agent_description or """This is a Quality Assurance/Quality Control agent that compares multiple outputs (e.g., reports, drafts, content) and selects the best one based on quality, accuracy, completeness, and relevance to the original query."""
    
    agent = CodeAgent(
        tools=[select_best_output],  # Add the selection tool
        model=model,
        additional_authorized_imports=["json"],
        name='qaqc_agent',
        description=description,
        max_steps=max_steps
    )
    
    # Define the custom system prompt if not provided
    custom_system_prompt = system_prompt or """You are a Quality Assurance/Quality Control agent responsible for comparing multiple outputs and selecting the best one.

When given multiple outputs to compare:

1. Carefully analyze each output for:
   - Accuracy of information
   - Completeness of coverage
   - Clarity and readability
   - Relevance to the original query
   - Logical structure and flow
   - Quality of writing and presentation

2. Create a detailed comparison table or rubric showing how each output performs on these criteria

3. Provide specific examples from each output to justify your assessment

4. Select the best output based on your analysis, clearly explaining why it's superior

5. If neither output is clearly superior overall, identify which parts of each are strongest and suggest how they could be combined

6. Always maintain objectivity in your assessment, focusing on measurable quality factors rather than subjective preferences

7. IMPORTANT: After completing your analysis, you MUST use the select_best_output tool to programmatically select the best output. This tool requires:
   - output_number: The number of the selected output (1 or 2)
   - output_text: The complete text of the selected output (copy and paste it exactly)

Your goal is to ensure that only the highest quality output moves forward in the process, improving the overall quality of the final result.

Do not modify either of the outputs you are given, only provide a comparison and selection of the best output.
"""

    # Apply custom templates with the custom system prompt
    apply_custom_agent_prompts(agent, custom_system_prompt)
    
    return agent 