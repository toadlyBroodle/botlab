#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
from typing import List, Dict, Any, Optional, Callable, Tuple

from dotenv import load_dotenv
from utils.telemetry import start_telemetry, suppress_litellm_logs
from manager.main import create_agent_by_type
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from qaqc.agents import run_qaqc_comparison

class AgentLoop:
    """A class that manages a loop of agent calls with state management."""
    
    def __init__(
        self,
        agent_sequence: List[str],
        max_iterations: int = 5,
        max_steps_per_agent: int = 15,
        max_retries: int = 3,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        use_custom_prompts: bool = False,
        enable_telemetry: bool = False,
        state_file: Optional[str] = None
    ):
        """Initialize the agent loop.
        
        Args:
            agent_sequence: List of agent types to call in sequence (e.g., ["researcher", "writer", "editor", "qaqc"])
            max_iterations: Maximum number of iterations through the entire sequence
            max_steps_per_agent: Maximum steps for each agent
            max_retries: Maximum retries for rate limiting
            model_id: The model ID to use
            model_info_path: Path to model info JSON file
            use_custom_prompts: Whether to use custom agent descriptions and prompts
            enable_telemetry: Whether to enable OpenTelemetry tracing
            state_file: Optional path to a file for persisting state between runs
        """
        self.agent_sequence = agent_sequence
        self.max_iterations = max_iterations
        self.max_steps_per_agent = max_steps_per_agent
        self.max_retries = max_retries
        self.model_id = model_id
        self.model_info_path = model_info_path
        self.use_custom_prompts = use_custom_prompts
        self.enable_telemetry = enable_telemetry
        self.state_file = state_file
        
        # Initialize state
        self.state = {
            "iteration": 0,
            "current_agent_index": 0,
            "results": {},
            "status": "initialized",
            "error": None,
            "start_time": time.time(),
            "last_updated": time.time()
        }
        
        # Load state from file if provided
        if state_file and os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                    print(f"Loaded state from {state_file}")
            except Exception as e:
                print(f"Error loading state from {state_file}: {e}")
        
        # Set up environment
        self._setup_environment()
        
        # Initialize agent configs
        self.agent_configs = self._get_default_agent_configs() if use_custom_prompts else {}
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
    
    def _setup_environment(self):
        """Set up environment variables, API keys, and telemetry."""
        load_dotenv()
        
        # Suppress LiteLLM logs
        suppress_litellm_logs()
        
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Start telemetry if enabled
        if self.enable_telemetry:
            tracer = start_telemetry(
                agent_name="agent_loop", 
                agent_type=f"agent_loop_with_{','.join(self.agent_sequence)}"
            )
    
    def _get_default_agent_configs(self):
        """Get default agent configurations.
        
        Returns:
            Dictionary with default agent configurations
        """
        return {
            "researcher_description": "Expert researcher with focus on scientific papers and academic sources",
            "researcher_prompt": "You are a meticulous researcher who prioritizes academic sources and provides comprehensive information with proper citations.",
            
            "writer_description": "Creative writer with journalistic style and clear explanations",
            "writer_prompt": "Write engaging content with a focus on clarity, accuracy, and reader engagement. Use a journalistic style that makes complex topics accessible.",
            
            "critic_description": "Detail-oriented editor with high standards for clarity and accuracy",
            "critic_prompt": "Evaluate writing for clarity, accuracy, engagement, and logical flow. Provide constructive feedback that improves the content without changing its voice.",
            
            "editor_description": "Skilled editor with focus on accuracy, clarity, and factual correctness",
            "editor_prompt": "Edit content to ensure factual accuracy while maintaining style and readability. Focus on improving clarity without changing the author's voice.",
            
            "fact_checker_description": "Thorough fact checker with attention to detail and source verification",
            "fact_checker_prompt": "Verify claims against reliable sources with precision. Identify potential inaccuracies and suggest corrections based on authoritative references."
        }
    
    def _initialize_agents(self):
        """Initialize all agents in the sequence."""
        for agent_type in self.agent_sequence:
            if agent_type not in self.agents:
                try:
                    agent = create_agent_by_type(
                        agent_type=agent_type,
                        max_steps=self.max_steps_per_agent,
                        model_id=self.model_id,
                        model_info_path=self.model_info_path,
                        max_retries=self.max_retries,
                        agent_configs=self.agent_configs
                    )
                    self.agents[agent_type] = agent
                    print(f"Initialized agent: {agent_type}")
                except Exception as e:
                    print(f"Error initializing agent {agent_type}: {e}")
                    self.state["error"] = f"Error initializing agent {agent_type}: {str(e)}"
                    self.state["status"] = "error"
    
    def _save_state(self):
        """Save the current state to a file if a state file is specified."""
        if self.state_file:
            try:
                # Update last_updated timestamp
                self.state["last_updated"] = time.time()
                
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
                print(f"Saved state to {self.state_file}")
            except Exception as e:
                print(f"Error saving state to {self.state_file}: {e}")
    
    def _format_prompt_for_agent(self, agent_type: str, query: str, previous_results: Dict[str, Any]) -> str:
        """Format a prompt for a specific agent type, incorporating previous results.
        
        Args:
            agent_type: The type of agent to format the prompt for
            query: The original query
            previous_results: Dictionary of results from previous agents
            
        Returns:
            Formatted prompt for the agent
        """
        prompt = f"Original query: {query}\n\n"
        
        # Add context from previous agents if available
        if previous_results:
            prompt += "Previous results:\n"
            for prev_agent, result in previous_results.items():
                if prev_agent != agent_type:  # Don't include this agent's own previous results
                    prompt += f"\n--- Results from {prev_agent} ---\n{result}\n"
        
        # Add specific instructions based on agent type
        if agent_type == "researcher":
            prompt += "\nYour task is to research this topic thoroughly and provide comprehensive information with proper citations."
        elif agent_type == "writer":
            prompt += "\nYour task is to write engaging content based on the research provided."
        elif agent_type == "editor":
            prompt += "\nYour task is to edit and fact-check the content, ensuring accuracy while maintaining style and readability."
        elif agent_type == "qaqc":
            # For QAQC agent, we'll handle this differently in the run method
            # using the run_qaqc_comparison function directly
            prompt = "QAQC agent will be handled separately"
        
        return prompt
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent loop for a given query.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary containing the final results and state information
        """
        print(f"Starting agent loop with query: {query}")
        print(f"Agent sequence: {' -> '.join(self.agent_sequence)}")
        
        self.state["status"] = "running"
        self.state["query"] = query
        
        try:
            # Continue from where we left off if resuming
            iteration = self.state["iteration"]
            agent_index = self.state["current_agent_index"]
            
            while iteration < self.max_iterations:
                print(f"\nIteration {iteration + 1}/{self.max_iterations}")
                
                # Process each agent in the sequence
                while agent_index < len(self.agent_sequence):
                    agent_type = self.agent_sequence[agent_index]
                    agent = self.agents.get(agent_type)
                    
                    if not agent:
                        error_msg = f"Agent {agent_type} not initialized"
                        print(f"Error: {error_msg}")
                        self.state["error"] = error_msg
                        self.state["status"] = "error"
                        self._save_state()
                        return self.state
                    
                    print(f"Running agent: {agent_type}")
                    
                    # Get previous results for context
                    previous_results = {k: v for k, v in self.state.get("results", {}).items()}
                    
                    # Format prompt for this agent
                    formatted_prompt = self._format_prompt_for_agent(agent_type, query, previous_results)
                    
                    # Run the agent
                    try:
                        start_time = time.time()
                        
                        # Special handling for QAQC agent
                        if agent_type == "qaqc":
                            # Only run QAQC if we have enough iterations
                            if iteration > 0:
                                # Get the previous agent in the sequence (the one before QAQC)
                                qaqc_index = self.agent_sequence.index("qaqc")
                                if qaqc_index > 0:
                                    previous_agent_type = self.agent_sequence[qaqc_index - 1]
                                else:
                                    previous_agent_type = self.agent_sequence[-1]  # Wrap around to the last agent
                                
                                # Get the current and previous outputs from the agent before QAQC
                                current_output = previous_results.get(f"{previous_agent_type}_{iteration}")
                                previous_output = previous_results.get(f"{previous_agent_type}_{iteration - 1}")
                                
                                if current_output and previous_output:
                                    # Create outputs dictionary for comparison
                                    outputs = {
                                        "Previous Iteration": previous_output,
                                        "Current Iteration": current_output
                                    }
                                    
                                    # Run the QAQC comparison
                                    selected_output, result, selected_name = run_qaqc_comparison(
                                        query=query,
                                        outputs=outputs,
                                        model=self.model,
                                        max_steps=self.max_steps_per_agent
                                    )
                                    
                                    # Update the result based on the selection
                                    if selected_name == "Previous Iteration":
                                        print("QAQC selected the previous iteration's output as better")
                                        
                                        # Replace the current iteration's result with the previous iteration's result
                                        self.state["results"][f"{previous_agent_type}_{iteration}"] = previous_output
                                        self.state["results"][previous_agent_type] = previous_output
                                        print(f"Updated {previous_agent_type} result with the better version from the previous iteration")
                                    else:
                                        print("QAQC selected the current iteration's output as better")
                                else:
                                    result = "Not enough outputs to compare yet. Please wait for more iterations."
                            else:
                                result = "Not enough iterations to compare outputs yet. Please wait for more iterations."
                        else:
                            # For all other agents, run normally
                            result = agent.run(formatted_prompt)
                        
                        end_time = time.time()
                        
                        # Store the result
                        if "results" not in self.state:
                            self.state["results"] = {}
                        
                        # Store with iteration number to keep history
                        result_key = f"{agent_type}_{iteration}"
                        self.state["results"][result_key] = result
                        
                        # Also store as the latest result for this agent type
                        self.state["results"][agent_type] = result
                        
                        print(f"Agent {agent_type} completed in {end_time - start_time:.2f} seconds")
                        print(f"Result length: {len(result)} characters")
                        
                        # Update state
                        self.state["current_agent_index"] = agent_index + 1
                        self._save_state()
                        
                    except Exception as e:
                        error_msg = f"Error running agent {agent_type}: {str(e)}"
                        print(f"Error: {error_msg}")
                        self.state["error"] = error_msg
                        self.state["status"] = "error"
                        self._save_state()
                        return self.state
                    
                    # Move to the next agent
                    agent_index += 1
                
                # Completed one full iteration
                iteration += 1
                self.state["iteration"] = iteration
                
                # Reset agent index for the next iteration
                agent_index = 0
                self.state["current_agent_index"] = agent_index
                
                # Save state after each iteration
                self._save_state()
                
                # Check if we should continue to the next iteration
                if iteration >= self.max_iterations:
                    break
            
            # Loop completed successfully
            self.state["status"] = "completed"
            self._save_state()
            
            return self.state
            
        except Exception as e:
            error_msg = f"Unexpected error in agent loop: {str(e)}"
            print(f"Error: {error_msg}")
            self.state["error"] = error_msg
            self.state["status"] = "error"
            self._save_state()
            return self.state

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a loop of agent calls with state management")
    
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument("--agent-sequence", type=str, default="researcher,writer,editor", 
                        help="Comma-separated list of agent types to call in sequence")
    parser.add_argument("--max-iterations", type=int, default=3, 
                        help="Maximum number of iterations through the entire sequence")
    parser.add_argument("--max-steps-per-agent", type=int, default=15, 
                        help="Maximum steps for each agent")
    parser.add_argument("--max-retries", type=int, default=3, 
                        help="Maximum retries for rate limiting")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", 
                        help="The model ID to use")
    parser.add_argument("--model-info-path", type=str, default="utils/gemini/gem_llm_info.json", 
                        help="Path to model info JSON file")
    parser.add_argument("--use-custom-prompts", action="store_true", 
                        help="Whether to use custom agent descriptions and prompts")
    parser.add_argument("--enable-telemetry", action="store_true", 
                        help="Whether to enable OpenTelemetry tracing")
    parser.add_argument("--state-file", type=str, default=None, 
                        help="Path to a file for persisting state between runs")
    
    return parser.parse_args()

def main():
    """Main entry point for the agent loop script."""
    args = parse_args()
    
    # Parse agent sequence
    agent_sequence = [agent_type.strip() for agent_type in args.agent_sequence.split(",")]
    
    # Initialize the agent loop
    agent_loop = AgentLoop(
        agent_sequence=agent_sequence,
        max_iterations=args.max_iterations,
        max_steps_per_agent=args.max_steps_per_agent,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        use_custom_prompts=args.use_custom_prompts,
        enable_telemetry=args.enable_telemetry,
        state_file=args.state_file
    )
    
    # Run the agent loop
    result = agent_loop.run(args.query)
    
    # Print the final result
    if result["status"] == "completed":
        print("\n=== Final Results ===")
        
        # Get the final result from the last agent in the sequence
        final_agent = agent_sequence[-1]
        final_result = result["results"].get(final_agent)
        
        if final_result:
            print(f"\n{final_result}")
        else:
            print("No final result available.")
    else:
        print(f"\nAgent loop ended with status: {result['status']}")
        if result.get("error"):
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main() 