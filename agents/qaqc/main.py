#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Optional, Callable, Dict, Tuple

from dotenv import load_dotenv
from utils.telemetry import start_telemetry, suppress_litellm_logs
from qaqc.agents import create_qaqc_agent, run_qaqc_comparison
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel

def setup_environment(enable_telemetry=False):
    """Set up environment variables, API keys, and telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
    """
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Start telemetry if enabled
    if enable_telemetry:
        tracer = start_telemetry(
            agent_name="qaqc_agent", 
            agent_type="qaqc"
        )

def initialize(
    enable_telemetry: bool = False,
    max_steps: int = 15,
    max_retries: int = 3,
    model_id: str = "gemini/gemini-2.0-flash",
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    agent_description: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> Callable[[str, Dict[str, str]], Tuple[str, str, str]]:
    """Initialize the QAQC agent system with optional telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        max_steps: Maximum number of steps for the agent
        max_retries: Maximum number of retry attempts for rate limiting
        model_id: The model ID to use
        model_info_path: Path to the model info JSON file
        agent_description: Optional custom description for the agent
        system_prompt: Optional custom system prompt for the agent
        
    Returns:
        A function that can process comparison requests through the QAQC agent
    """
    
    # Set up environment and telemetry
    setup_environment(enable_telemetry=enable_telemetry)
    
    # Create a rate-limited model with model-specific rate limits
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        max_retries=max_retries,
        model_info_path=model_info_path
    )
    
    def compare_outputs(query: str, outputs: Dict[str, str]) -> Tuple[str, str, str]:
        """Runs a comparison of multiple outputs through the QAQC agent
        
        Args:
            query: The original query
            outputs: Dictionary of outputs to compare (key: output_name, value: output_text)
            
        Returns:
            Tuple containing:
            - The selected output text
            - The full QAQC analysis
            - The name of the selected output
        """
        return run_qaqc_comparison(
            query=query,
            outputs=outputs,
            model=model,
            max_steps=max_steps,
            agent_description=agent_description,
            system_prompt=system_prompt
        )
    
    # Wrap with traced decorator if telemetry is enabled
    if enable_telemetry:
        from utils.telemetry import traced
        compare_outputs = traced(
            span_name="qaqc.compare_outputs",
            attributes={"agent.type": "qaqc"}
        )(compare_outputs)
        
    return compare_outputs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the QAQC agent to compare outputs")
    
    parser.add_argument("--output1", type=str, required=True, help="First output to compare")
    parser.add_argument("--output2", type=str, required=True, help="Second output to compare")
    parser.add_argument("--output1-name", type=str, default="Previous Iteration", help="Name for the first output")
    parser.add_argument("--output2-name", type=str, default="Current Iteration", help="Name for the second output")
    parser.add_argument("--query", type=str, required=True, help="Original query for context")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum steps for the agent")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="The model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--enable-telemetry", action="store_true", help="Whether to enable OpenTelemetry tracing")
    parser.add_argument("--output-selected-only", action="store_true", help="Only output the selected text, not the analysis")
    
    return parser.parse_args()

def main():
    """Main entry point for the QAQC agent."""
    args = parse_args()
    
    # Initialize the QAQC agent
    compare_outputs = initialize(
        enable_telemetry=args.enable_telemetry,
        max_steps=args.max_steps,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path
    )
    
    # Create outputs dictionary
    outputs = {
        args.output1_name: args.output1,
        args.output2_name: args.output2
    }
    
    # Run the comparison
    selected_output, analysis, selected_name = compare_outputs(args.query, outputs)
    
    # Print the result
    if args.output_selected_only:
        print(selected_output)
    else:
        print("\n=== QAQC Agent Comparison Result ===\n")
        print(analysis)
        print("\n=== Selected Output ===\n")
        print(f"Selected: {selected_name}")
        print(selected_output)

if __name__ == "__main__":
    main() 