#!/usr/bin/env python3
"""
Example usage of the QAQCAgent class.

This example shows how to create and use a QAQCAgent instance directly.
It also provides a command-line interface for comparing multiple outputs.

Usage:
    python -m agents.qaqc.example --outputs output1.txt output2.txt --query "Original query"
"""

import os
import argparse
from dotenv import load_dotenv
from agents.utils.telemetry import suppress_litellm_logs
from agents.qaqc.agents import QAQCAgent

def setup_basic_environment():
    """Set up basic environment for the example"""
    # Load .env from root directory
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

def run_example(output1=None, output2=None, query=None, max_steps=10, model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                agent_description=None, agent_prompt=None):
    """Run a QAQC comparison using the QAQCAgent class
    
    Args:
        output1: The first output to compare
        output2: The second output to compare
        query: Optional original query for context
        max_steps: Maximum number of steps for the agent
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        agent_description: Optional custom description for the agent
        agent_prompt: Optional custom system prompt for the agent
        
    Returns:
        The comparison result from the QAQC agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Use default outputs if none provided
    if output1 is None:
        output1 = """# Quantum Computing Advancements

Quantum computing has seen significant progress in recent years. IBM's quantum computer reached 127 qubits in 2021, while Google achieved quantum supremacy in 2019 with their 53-qubit Sycamore processor. Quantum computers use quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits.

The potential applications include cryptography, drug discovery, and optimization problems. Companies like Microsoft, Amazon, and Intel are also investing heavily in quantum research. D-Wave Systems has developed quantum annealing machines with over 5000 qubits, though these are specialized for optimization problems rather than general-purpose quantum computing.

Researchers at MIT recently developed a new quantum algorithm that could exponentially speed up certain types of calculations. The future of quantum computing looks promising, with experts predicting practical applications within the next decade."""
    
    if output2 is None:
        output2 = """# Recent Advances in Quantum Computing Technology

Quantum computing has made remarkable strides in the past few years. In 2021, IBM announced their Eagle processor with 127 qubits, a significant milestone in the field. Google demonstrated quantum supremacy back in 2019 using their 53-qubit Sycamore processor, completing a calculation in minutes that would take classical supercomputers thousands of years.

Unlike classical computers that use bits (0 or 1), quantum computers leverage qubits that can exist in superposition states, enabling them to process complex problems more efficiently. This technology has promising applications across multiple domains:

1. **Cryptography**: Developing post-quantum cryptography to protect against quantum attacks
2. **Drug Discovery**: Simulating molecular structures for faster pharmaceutical development
3. **Optimization Problems**: Solving complex logistics and supply chain challenges

Major tech companies including Microsoft (Q#), Amazon (Braket), and Intel are heavily investing in quantum research and development. D-Wave Systems has taken a different approach with quantum annealing machines containing over 5000 qubits, though these are designed specifically for optimization problems.

MIT researchers recently published a breakthrough quantum algorithm that demonstrates exponential speedup for certain computational tasks. Industry experts project that practical, commercially viable quantum applications will emerge within the next 5-10 years as the technology continues to mature."""
    
    # Use default description if none provided
    if agent_description is None:
        agent_description = "Specialized in comparing outputs and selecting the best one based on quality, accuracy, and completeness."
    
    print(f"Creating QAQC agent with max_steps={max_steps}")
    
    # Create the QAQC agent
    qaqc = QAQCAgent(
        max_steps=max_steps,
        agent_description=agent_description,
        system_prompt=agent_prompt,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )
    
    print("Comparing outputs...")
    print("=" * 80)
    
    # Run the comparison and get the result
    result = qaqc.compare_outputs([output1, output2], query)
    
    print("=" * 80)
    print("Comparison complete! The comparison has been saved.")
    
    return result

def read_file_content(file_path):
    """Read content from a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        The content of the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the QAQCAgent to compare outputs.")
    parser.add_argument("--outputs", nargs='+', help="Paths to the output files or content directly")
    parser.add_argument("--query", type=str, help="Optional original query for context")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--agent-description", type=str, help="Custom description for the QAQC agent")
    parser.add_argument("--agent-prompt", type=str, help="Custom system prompt for the QAQC agent")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Process the outputs
    outputs = args.outputs if args.outputs else []
    
    # Read file contents if they are files
    processed_outputs = []
    for output_path in outputs:
        if os.path.isfile(output_path):
            content = read_file_content(output_path)
            if content:
                processed_outputs.append(content)
        else:
            processed_outputs.append(output_path)
    
    if len(processed_outputs) < 2:
        raise ValueError("Please provide at least two outputs to compare")
    elif len(processed_outputs) > 2:
        print("Warning: Only the first two outputs will be compared")

    # Get the first two outputs (QAQCAgent currently only supports comparing two outputs)
    output1 = processed_outputs[0] if len(processed_outputs) > 0 else None
    output2 = processed_outputs[1] if len(processed_outputs) > 1 else None
    
    run_example(
        output1=output1,
        output2=output2,
        query=args.query,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        agent_description=args.agent_description,
        agent_prompt=args.agent_prompt
    ) 