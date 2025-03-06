#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime

from dotenv import load_dotenv
from utils.telemetry import start_telemetry, suppress_litellm_logs
from manager.main import create_agent_by_type
from utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from qaqc.main import initialize as initialize_qaqc
from utils.file_manager.file_manager import FileManager

class AgentLoop:
    """A class that manages a loop of agent calls with state management."""
    
    def __init__(
        self,
        agent_sequence: List[str],
        max_iterations: int = 5,
        max_steps_per_agent: Union[int, str] = 5,
        max_retries: int = 3,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "utils/gemini/gem_llm_info.json",
        use_custom_prompts: bool = False,
        enable_telemetry: bool = False,
        state_file: Optional[str] = None,
        load_state: bool = False
    ):
        """Initialize the agent loop.
        
        Args:
            agent_sequence: List of agent types to call in sequence (e.g., ["researcher", "writer", "editor", "qaqc"])
            max_iterations: Maximum number of iterations through the entire sequence
            max_steps_per_agent: Maximum steps for each agent. Can be either:
                - An integer (same value for all agents)
                - A comma-separated string (e.g., "3,3,9,1" for different values per agent)
            max_retries: Maximum retries for rate limiting
            model_id: The model ID to use
            model_info_path: Path to model info JSON file
            use_custom_prompts: Whether to use custom agent descriptions and prompts
            enable_telemetry: Whether to enable OpenTelemetry tracing
            state_file: Optional path to a file for persisting state between runs
            load_state: Whether to load state from state_file if it exists (default: False)
        """
        self.agent_sequence = agent_sequence
        self.max_iterations = max_iterations
        
        # Parse max_steps_per_agent
        if isinstance(max_steps_per_agent, str) and "," in max_steps_per_agent:
            # Parse comma-separated list of max steps
            steps_list = [int(steps.strip()) for steps in max_steps_per_agent.split(",")]
            self.max_steps_per_agent_dict = {}
            
            # Map steps to agents
            for i, agent_type in enumerate(agent_sequence):
                if i < len(steps_list):
                    self.max_steps_per_agent_dict[agent_type] = steps_list[i]
                else:
                    # Use the last value for any remaining agents
                    self.max_steps_per_agent_dict[agent_type] = steps_list[-1]
            
            # Store the default value for any agents not in the sequence
            self.max_steps_per_agent = steps_list[-1] if steps_list else 5
        else:
            # Use the same value for all agents
            self.max_steps_per_agent = int(max_steps_per_agent)
            self.max_steps_per_agent_dict = {agent_type: self.max_steps_per_agent for agent_type in agent_sequence}
        
        self.max_retries = max_retries
        self.model_id = model_id
        self.model_info_path = model_info_path
        self.use_custom_prompts = use_custom_prompts
        self.enable_telemetry = enable_telemetry
        self.state_file = state_file
        
        # Initialize state
        self.state = {
            "status": "initialized",
            "current_iteration": 0,
            "current_agent": 0,
            "results": {},
            "error": None,
            "last_updated": time.time()
        }
        
        # Initialize the file manager
        self.file_manager = FileManager()
        
        # Load state from file if provided and load_state is True
        if load_state and state_file and os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    loaded_state = json.load(f)
                    self.state.update(loaded_state)
                    
                    # Ensure critical state variables are valid
                    if self.state.get("current_agent") is None:
                        self.state["current_agent"] = 0
                        
                    if self.state.get("current_iteration") is None:
                        self.state["current_iteration"] = 0
                        
                    if self.state.get("results") is None:
                        self.state["results"] = {}
                        
                    # Add iteration key for compatibility with agent_loop_example.py
                    self.state["iteration"] = self.state["current_iteration"]
                    
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
                        max_steps=self.max_steps_per_agent_dict.get(agent_type, self.max_steps_per_agent),
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
                    # Add iteration key for compatibility with agent_loop_example.py
                    self.state["iteration"] = self.state["current_iteration"]
                    self._save_state()
                    return self.state
    
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
    
    def _get_agent_file_type(self, agent_type: str) -> str:
        """Get the file type associated with an agent type.
        
        Args:
            agent_type: The type of agent
            
        Returns:
            The file type (draft, report, resource, etc.)
        """
        # Map agent types to file types
        agent_file_type_map = {
            "researcher": "report",
            "writer": "draft",
            "editor": "draft",
            "critic": "report",
            "fact_checker": "report",
            "qaqc": "report"
        }
        
        return agent_file_type_map.get(agent_type, "resource")
    
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
            
            # Filter out results that include iteration numbers (except the latest for each agent type)
            # This ensures we only include the most recent/selected outputs
            latest_results = {}
            for prev_agent, result in previous_results.items():
                # Skip this agent's own previous results
                if prev_agent == agent_type:
                    continue
                    
                # Check if this is an iteration-specific result (e.g., "researcher_0")
                if "_" in prev_agent:
                    base_agent, iteration = prev_agent.rsplit("_", 1)
                    # Only keep if we don't have a non-iteration version of this agent's result
                    if base_agent not in previous_results:
                        latest_results[base_agent] = result
                else:
                    # This is the latest result for this agent type
                    latest_results[prev_agent] = result
            
            # Add the filtered results to the prompt
            for prev_agent, result in latest_results.items():
                prompt += f"\n--- Results from {prev_agent} ---\n{result}\n"
        
        # Try to load the latest file for the previous agent in the sequence
        current_agent_index = self.agent_sequence.index(agent_type)
        if current_agent_index > 0:
            prev_agent_type = self.agent_sequence[current_agent_index - 1]
            prev_file_type = self._get_agent_file_type(prev_agent_type)
            
            # List files of the previous agent's type, sorted by creation date (newest first)
            prev_files = self.file_manager.list_files(file_type=prev_file_type)
            
            if prev_files:
                # Get the most recent file
                latest_file = prev_files[0]
                try:
                    # Get the file content
                    file_data = self.file_manager.get_file(latest_file["file_id"])
                    file_content = file_data["content"]
                    file_title = latest_file.get("title", "Untitled")
                    
                    # Add to prompt
                    prompt += f"\n--- Latest {prev_file_type.capitalize()} from {prev_agent_type}: {file_title} ---\n"
                    prompt += f"{file_content}\n"
                    
                    print(f"Added latest {prev_file_type} from {prev_agent_type} to prompt")
                except Exception as e:
                    print(f"Error loading file: {e}")
        
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
        
        # Reset status if it was previously completed or if we're loading from state
        if self.state["status"] in ["completed", "initialized"]:
            print(f"Previous run status was '{self.state['status']}'. Starting a new iteration.")
            # Keep the results but reset the agent index to start a new iteration
            self.state["status"] = "running"
            self.state["current_agent"] = 0
            
            # Always preserve previous results when loading state
            # This ensures we can build upon previous work
            
            # Reset iteration counter to 0 to run full set of iterations
            self.state["current_iteration"] = 0
        
        self.state["status"] = "running"
        self.state["query"] = query
        
        try:
            # Continue from where we left off if resuming
            iteration = self.state["current_iteration"]
            agent_index = self.state["current_agent"]
            
            print(f"Starting from iteration {iteration + 1}, agent index {agent_index}")
            
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
                        # Add iteration key for compatibility with agent_loop_example.py
                        self.state["iteration"] = self.state["current_iteration"]
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
                            # Get the previous agent in the sequence (the one before QAQC)
                            qaqc_index = self.agent_sequence.index("qaqc")
                            if qaqc_index > 0:
                                previous_agent_type = self.agent_sequence[qaqc_index - 1]
                            else:
                                previous_agent_type = self.agent_sequence[-1]  # Wrap around to the last agent
                            
                            # Get the current and previous outputs from the agent before QAQC
                            current_output = previous_results.get(f"{previous_agent_type}_{iteration}")
                            previous_output = previous_results.get(f"{previous_agent_type}_{iteration - 1}")
                            
                            # Create a dictionary of outputs to compare
                            outputs_to_compare = {}
                            
                            # Only add outputs that exist
                            if previous_output:
                                outputs_to_compare["Previous Iteration"] = previous_output
                            if current_output:
                                outputs_to_compare["Current Iteration"] = current_output
                            
                            try:
                                # Initialize the QAQC comparison function
                                compare_outputs = initialize_qaqc(
                                    max_steps=self.max_steps_per_agent_dict.get(previous_agent_type, self.max_steps_per_agent),
                                    max_retries=self.max_retries,
                                    model_id=self.model_id,
                                    model_info_path=self.model_info_path,
                                    enable_telemetry=self.enable_telemetry
                                )
                                
                                # Run the comparison
                                selected_output, result, selected_name = compare_outputs(query, outputs_to_compare)
                                
                                # If we have two outputs and a comparison was made
                                if "Previous Iteration" in outputs_to_compare and "Current Iteration" in outputs_to_compare and selected_name:
                                    # Update the result based on the selection
                                    if selected_name == "Previous Iteration":
                                        print("QAQC selected the previous iteration's output as better")
                                        
                                        # Replace the current iteration's result with the previous iteration's result
                                        self.state["results"][f"{previous_agent_type}_{iteration}"] = selected_output
                                        self.state["results"][previous_agent_type] = selected_output
                                        
                                        print(f"Updated {previous_agent_type} result with the better version from the previous iteration")
                                    else:
                                        print("QAQC selected the current iteration's output as better")
                                        
                                        # Make sure the current output is stored correctly
                                        self.state["results"][f"{previous_agent_type}_{iteration}"] = selected_output
                                        self.state["results"][previous_agent_type] = selected_output
                                        
                                        # Remove the previous iteration's result since it was not selected
                                        # This ensures it won't be included in future prompts
                                        if f"{previous_agent_type}_{iteration-1}" in self.state["results"]:
                                            del self.state["results"][f"{previous_agent_type}_{iteration-1}"]
                                else:
                                    # Just log the result
                                    print(f"QAQC result: {result}")
                                    
                                    # If we have a single output, make sure it's stored correctly
                                    if selected_output and selected_name and previous_agent_type:
                                        self.state["results"][f"{previous_agent_type}_{iteration}"] = selected_output
                                        self.state["results"][previous_agent_type] = selected_output
                            except Exception as e:
                                result = f"Error in QAQC comparison: {str(e)}"
                                print(result)
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
                        self.state["current_agent"] = agent_index + 1
                        self._save_state()
                        
                    except Exception as e:
                        error_msg = f"Error running agent {agent_type}: {str(e)}"
                        print(f"Error: {error_msg}")
                        self.state["error"] = error_msg
                        self.state["status"] = "error"
                        # Add iteration key for compatibility with agent_loop_example.py
                        self.state["iteration"] = self.state["current_iteration"]
                        self._save_state()
                        return self.state
                    
                    # Move to the next agent
                    agent_index += 1
                
                # Completed one full iteration
                iteration += 1
                self.state["current_iteration"] = iteration
                
                # Reset agent index for the next iteration
                agent_index = 0
                self.state["current_agent"] = agent_index
                
                # Save state after each iteration
                self._save_state()
            
            # Loop completed successfully
            self.state["status"] = "completed"
            # Add iteration key for compatibility with agent_loop_example.py
            self.state["iteration"] = self.state["current_iteration"]
            self._save_state()
            
            return self.state
            
        except Exception as e:
            error_msg = f"Unexpected error in agent loop: {str(e)}"
            print(f"Error: {error_msg}")
            self.state["error"] = error_msg
            self.state["status"] = "error"
            # Add iteration key for compatibility with agent_loop_example.py
            self.state["iteration"] = self.state["current_iteration"]
            self._save_state()
            return self.state

def main():
    """Main entry point for the agent loop."""
    args = parse_args()
    
    print("=== Agent Loop ===")
    print(f"Query: {args.query}")
    print(f"Agent Sequence: {args.agent_sequence}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Max Steps Per Agent: {args.max_steps_per_agent}")
    print(f"Max Retries: {args.max_retries}")
    print(f"Model ID: {args.model_id}")
    print(f"Model Info Path: {args.model_info_path}")
    print(f"Use Custom Prompts: {args.use_custom_prompts}")
    print(f"Enable Telemetry: {args.enable_telemetry}")
    print(f"State File: {args.state_file}")
    print(f"Load State: {args.load_state}")
    print()
    
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
        state_file=args.state_file,
        load_state=args.load_state
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a loop of agent calls with state management")
    
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument("--agent-sequence", type=str, default="researcher,writer,editor,qaqc", 
                        help="Comma-separated list of agent types to call in sequence")
    parser.add_argument("--max-iterations", type=int, default=3, 
                        help="Maximum number of iterations through the entire sequence")
    parser.add_argument("--max-steps-per-agent", type=str, default="3,3,9,1", 
                        help="Maximum steps for each agent. Can be either: "
                             "- An integer (same value for all agents) "
                             "- A comma-separated string (e.g., '3,3,9,1' for different values per agent)")
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
    parser.add_argument("--state-file", type=str, default="../shared_data/logs/agent_loop_state.json", 
                        help="Path to a file for persisting state between runs")
    parser.add_argument("--load-state", action="store_true", 
                        help="Whether to load state from state_file if it exists (default: False)")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 