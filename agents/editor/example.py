#!/usr/bin/env python3
"""
Example usage of the EditorAgent and FactCheckerAgent classes.

This example shows how to create and use an EditorAgent and FactCheckerAgent directly.
It also provides a command-line interface for editing and fact-checking content.

Usage:
    python -m agents.editor.example --content "Content to edit and fact check"
"""

import os
import argparse
from dotenv import load_dotenv
from agents.utils.telemetry import suppress_litellm_logs
from agents.editor.agents import EditorAgent, FactCheckerAgent

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

def run_example(content=None, max_steps=50, model_id="gemini/gemini-2.0-flash", 
                model_info_path="agents/utils/gemini/gem_llm_info.json",
                base_wait_time=2.0, max_retries=3,
                editor_description=None, fact_checker_description=None,
                editor_prompt=None, fact_checker_prompt=None):
    """Run an editing task using the EditorAgent and FactCheckerAgent classes
    
    Args:
        content: The content to edit
        max_steps: Maximum number of steps for the editor agent
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        base_wait_time: Base wait time for rate limiting
        max_retries: Maximum retries for rate limiting
        editor_description: Optional custom description for the editor agent
        fact_checker_description: Optional custom description for the fact checker agent
        editor_prompt: Optional custom system prompt for the editor agent
        fact_checker_prompt: Optional custom system prompt for the fact checker agent
        
    Returns:
        The edited content from the editor agent
    """
    # Set up environment
    setup_basic_environment()
    
    # Use default descriptions if none provided
    if editor_description is None:
        editor_description = "Specialized in editing and improving content while ensuring factual accuracy."
    
    if fact_checker_description is None:
        fact_checker_description = "Specialized in verifying claims and providing detailed accuracy assessments."
    
    print(f"Creating editor agent with max_steps={max_steps}")
    
    # Create the editor agent (which will create its own fact checker agent)
    editor = EditorAgent(
        max_steps=max_steps,
        agent_description=editor_description,
        fact_checker_description=fact_checker_description,
        system_prompt=editor_prompt,
        fact_checker_prompt=fact_checker_prompt,
        model_id=model_id,
        model_info_path=model_info_path,
        base_wait_time=base_wait_time,
        max_retries=max_retries
    )
    
    # Use default content if none provided
    if content is None:
        content = """# Quantum Computing Advancements

Quantum computing has seen significant progress in recent years. IBM's quantum computer reached 127 qubits in 2021, while Google achieved quantum supremacy in 2019 with their 53-qubit Sycamore processor. Quantum computers use quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits.

The potential applications include cryptography, drug discovery, and optimization problems. Companies like Microsoft, Amazon, and Intel are also investing heavily in quantum research. D-Wave Systems has developed quantum annealing machines with over 5000 qubits, though these are specialized for optimization problems rather than general-purpose quantum computing.

Researchers at MIT recently developed a new quantum algorithm that could exponentially speed up certain types of calculations. The future of quantum computing looks promising, with experts predicting practical applications within the next decade."""
    
    print(f"Running editing task on content of length {len(content)}")
    print("=" * 80)
    
    # Run the editing task and get the result
    result = editor.edit_content(content)
    
    print("=" * 80)
    print("Editing complete! The edited content has been saved.")
    
    return result

def parse_arguments():
    """Parse command-line arguments
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run the EditorAgent with content to edit.")
    parser.add_argument("--content", type=str, 
                        help="The content to edit (if not provided, a default example will be used)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of steps")
    parser.add_argument("--base-wait-time", type=float, default=2.0, help="Base wait time for rate limiting")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID to use")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--editor-description", type=str, help="Custom description for the editor agent")
    parser.add_argument("--fact-checker-description", type=str, help="Custom description for the fact checker agent")
    parser.add_argument("--editor-prompt", type=str, help="Custom system prompt for the editor agent")
    parser.add_argument("--fact-checker-prompt", type=str, help="Custom system prompt for the fact checker agent")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    run_example(
        content=args.content,
        max_steps=args.max_steps,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        base_wait_time=args.base_wait_time,
        max_retries=args.max_retries,
        editor_description=args.editor_description,
        fact_checker_description=args.fact_checker_description,
        editor_prompt=args.editor_prompt,
        fact_checker_prompt=args.fact_checker_prompt
    ) 