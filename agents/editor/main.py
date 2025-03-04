#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from typing import Optional
from utils.telemetry import start_telemetry, suppress_litellm_logs
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from editor.agents import create_editor_agent, create_fact_checker_agent
from editor.tools import EDITS_DIR
def setup_environment(enable_telemetry=False, agent_name=None, agent_type=None):
    """Set up environment variables, API keys, and telemetry
    
    Args:
        enable_telemetry: Whether to enable OpenTelemetry tracing
        agent_name: Optional name for the agent in telemetry
        agent_type: Optional type of the agent in telemetry
    """
    # Ensure edits directory exists
    os.makedirs(EDITS_DIR, exist_ok=True)
    
    # Load .env from root directory
    load_dotenv()
    
    # Suppress LiteLLM logs
    suppress_litellm_logs()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Start telemetry if enabled
    if enable_telemetry:
        from utils.telemetry import start_telemetry, traced
        tracer = start_telemetry(
            agent_name=agent_name or "editor_system", 
            agent_type=agent_type or "editor"
        )

def initialize(
    max_steps: int = None,
    model_info_path: str = "utils/gemini/gem_llm_info.json",
    model_id: str = "gemini/gemini-2.0-flash",
    enable_telemetry: bool = False,
    editor_description: Optional[str] = None,
    fact_checker_description: Optional[str] = None,
    editor_prompt: Optional[str] = None,
    fact_checker_prompt: Optional[str] = None
):
    """Initialize the editor-fact checker system
    
    Args:
        max_steps: Maximum number of steps for the editor agent
        model_id: LLM model to use
        enable_telemetry: Whether to enable OpenTelemetry tracing
        editor_description: Optional additional description for the editor agent
        fact_checker_description: Optional additional description for the fact checker agent
        editor_prompt: Optional custom system prompt for the editor agent
        fact_checker_prompt: Optional custom system prompt for the fact checker agent
        
    Returns:
        A function that can process editing tasks
    """
    
    # Set up environment and telemetry for editor
    setup_environment(
        enable_telemetry=enable_telemetry,
        agent_name="editor_system",
        agent_type="editor"
    )
    
    model = RateLimitedLiteLLMModel(
        model_id=model_id,
        model_info_path=model_info_path
    )
    
    # Set default max_steps if None
    fact_checker_max_steps = 30 if max_steps is None else max_steps
    editor_max_steps = 50 if max_steps is None else max_steps
    
    # Set up telemetry for fact checker if enabled
    if enable_telemetry:
        setup_environment(
            enable_telemetry=True,
            agent_name="fact_checker_agent",
            agent_type="fact_checker"
        )
    
    # Create agents in the right order - fact checker first, then editor that manages fact checker
    fact_checker = create_fact_checker_agent(
        model=model,
        max_steps=fact_checker_max_steps,
        agent_description=fact_checker_description,
        system_prompt=fact_checker_prompt
    )
    
    editor = create_editor_agent(
        model=model,
        fact_checker=fact_checker,
        max_steps=editor_max_steps,
        agent_description=editor_description,
        system_prompt=editor_prompt
    )
    
    def run_editing_task(content: str, task: Optional[str] = None) -> str:
        """Run an editing task through the editor-fact checker system
        
        Args:
            content: The content to edit and fact check
            task: Optional specific editing task/instructions
            
        Returns:
            The edited and fact-checked content
        """
        # Construct the prompt combining content and task
        if task:
            prompt = f"Task: {task}\n\nContent to edit:\n{content}"
        else:
            prompt = f"Edit and fact check this content:\n{content}"
            
        # Run the editor agent with the prompt
        result = editor.run(prompt)
        return result
    
    # Wrap with traced decorator if telemetry is enabled
    if enable_telemetry:
        from utils.telemetry import traced
        run_editing_task = traced(
            span_name="editor.run_task",
            attributes={"agent.type": "editor"}
        )(run_editing_task)
        
    return run_editing_task

def main():
    """Main entry point when run directly"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the editor-fact checker system with content to edit.")
    parser.add_argument("--content", type=str, 
                       default="The first computer was invented by Charles Babbage in 1822. It could perform complex calculations using steam power.", help="The content to edit and fact check")
    parser.add_argument("--task", type=str, default=None, help="Optional specific editing task/instructions")
    parser.add_argument("--enable-telemetry", action="store_true", help="Enable telemetry")
    parser.add_argument("--max-steps", type=int, help="Maximum number of steps")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="LLM model to use")
    parser.add_argument("--editor-description", type=str, default=None, help="Custom description for the editor agent")
    parser.add_argument("--fact-checker-description", type=str, default=None, help="Custom description for the fact checker agent")
    parser.add_argument("--editor-prompt", type=str, default=None, help="Custom system prompt for the editor agent")
    parser.add_argument("--fact-checker-prompt", type=str, default=None, help="Custom system prompt for the fact checker agent")
    
    args = parser.parse_args()
    
    # Initialize the editor-fact checker system with parameters from command line
    run_editing_task = initialize(
        max_steps=args.max_steps,
        model_id=args.model_id,
        enable_telemetry=args.enable_telemetry,
        editor_description=args.editor_description,
        fact_checker_description=args.fact_checker_description,
        editor_prompt=args.editor_prompt,
        fact_checker_prompt=args.fact_checker_prompt
    )
    
    # Run the editing task
    result = run_editing_task(args.content, args.task)
    
    # Print the result
    print("\nEdited and Fact-Checked Content:")
    print(result)
    
    return result

if __name__ == "__main__":
    main() 