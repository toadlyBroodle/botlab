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

def run_qaqc_comparison(
    query: str,
    outputs: Dict[str, str],
    model: RateLimitedLiteLLMModel,
    max_steps: int = 15,
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> Tuple[str, str, str]:
    """Run a QAQC comparison on multiple outputs and return the selected output
    
    Args:
        query: The original query
        outputs: Dictionary of outputs to compare (key: output_name, value: output_text)
        model: The model to use for the agent
        max_steps: Maximum steps for the agent
        agent_description: Optional custom description for the agent
        system_prompt: Optional custom system prompt for the agent
        
    Returns:
        Tuple containing:
        - The selected output text
        - The full QAQC analysis
        - The name of the selected output
    """
    # Create the QAQC agent
    qaqc_agent = create_qaqc_agent(
        model=model,
        max_steps=max_steps,
        agent_description=agent_description,
        system_prompt=system_prompt
    )
    
    # Format the comparison request
    output_names = list(outputs.keys())
    
    if len(output_names) < 2:
        raise ValueError("At least two outputs are required for comparison")
    
    # For now, we only support comparing two outputs
    if len(output_names) > 2:
        print("Warning: More than two outputs provided. Only the first two will be compared.")
    
    output1_name = output_names[0]
    output2_name = output_names[1]
    output1_text = outputs[output1_name]
    output2_text = outputs[output2_name]
    
    comparison_request = f"""
Original Query: {query}

I need you to compare the following two outputs and select the best one based on quality, accuracy, completeness, and relevance to the original query.

OUTPUT 1 ({output1_name}):
{output1_text}

OUTPUT 2 ({output2_name}):
{output2_text}

Please analyze both outputs carefully and provide a detailed comparison. Then select the best output to move forward with by using the select_best_output tool.
"""
    
    # Run the comparison
    result = qaqc_agent.run(comparison_request)
    
    # Extract the selection from the agent's tool calls
    selected_output = ""
    selected_name = output2_name  # Default to the second output
    
    # Look for tool calls in the agent's memory steps
    if hasattr(qaqc_agent, 'memory') and hasattr(qaqc_agent.memory, 'steps'):
        for step in qaqc_agent.memory.steps:
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tool_call in step.tool_calls:
                    if tool_call.name == "select_best_output" and tool_call.result.get("success", False):
                        output_number = tool_call.result.get("selected_output_number")
                        selected_output = tool_call.result.get("selected_output_text", "")
                        
                        if output_number == 1:
                            selected_name = output1_name
                        elif output_number == 2:
                            selected_name = output2_name
    
    # If no selected output was found in the tool calls, use the one from the name
    if not selected_output:
        selected_output = output1_text if selected_name == output1_name else output2_text
    
    return selected_output, result, selected_name 