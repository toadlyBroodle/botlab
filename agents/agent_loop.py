#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime

from dotenv import load_dotenv
from .utils.telemetry import start_telemetry, suppress_litellm_logs
from .utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from .utils.file_manager.file_manager import FileManager
from .utils.agents.tools import load_file

# Import the agent classes
from .researcher.agents import ResearcherAgent
from .writer_critic.agents import WriterAgent, CriticAgent
from .editor.agents import EditorAgent, FactCheckerAgent
from .qaqc.agents import QAQCAgent
from .user_feedback.agents import UserFeedbackAgent
from .user_feedback.tools import FB_AGENT_USER

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
        load_state: bool = False,
        agent_configs: Optional[Dict[str, Any]] = None,
        agent_contexts: Optional[Dict[str, Dict[str, Any]]] = None
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
            agent_configs: Optional dictionary containing custom configuration for agents
            agent_contexts: Optional dictionary containing context data for specific agents
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
        self.agent_contexts = agent_contexts or {}
        
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
        
        # Initialize agent configs with provided configs or defaults
        if use_custom_prompts:
            self.agent_configs = self._get_default_agent_configs()
            # Update with any provided configs
            if agent_configs:
                self.agent_configs.update(agent_configs)
        else:
            self.agent_configs = agent_configs or {}
        
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
            "fact_checker_prompt": "Verify claims against reliable sources with precision. Identify potential inaccuracies and suggest corrections based on authoritative references.",
            
            "qaqc_description": "Quality assurance specialist with focus on comparing outputs and selecting the best one",
            "qaqc_prompt": "Compare outputs based on quality, accuracy, completeness, and relevance to the original query. Select the best output and explain your reasoning.",
            
            "user_email": "rob@botlab.dev",
            "report_frequency": 1,
            "user_feedback_description": f"User feedback agent that uses the {FB_AGENT_USER} system user for email communication",
            "user_feedback_prompt": f"You are a user feedback agent that communicates with users via email. You use the {FB_AGENT_USER} system user to send and receive emails. Your goal is to keep users informed about the progress of agent loops and to process their feedback and commands."
        }
    
    def _initialize_agents(self):
        """Initialize all agents in the sequence."""
        for agent_type in self.agent_sequence:
            if agent_type not in self.agents:
                try:
                    # Get the max steps for this agent
                    max_steps = self.max_steps_per_agent_dict.get(agent_type, self.max_steps_per_agent)
                    
                    # Get any specific context for this agent
                    agent_context = self.agent_contexts.get(agent_type, {}).copy()
                    
                    # Create the agent based on its type
                    if agent_type.lower() == 'researcher':
                        # Extract special parameters that shouldn't be directly passed to the agent constructor
                        if 'research_parameters' in agent_context:
                            self.research_parameters = agent_context.pop('research_parameters')
                        if 'original_row' in agent_context:
                            self.original_row = agent_context.pop('original_row')
                        
                        agent_instance = ResearcherAgent(
                            max_steps=max_steps,
                            researcher_description=self.agent_configs.get('researcher_description'),
                            researcher_prompt=self.agent_configs.get('researcher_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                        
                    elif agent_type.lower() == 'writer':
                        agent_instance = WriterAgent(
                            max_steps=max_steps,
                            agent_description=self.agent_configs.get('writer_description'),
                            system_prompt=self.agent_configs.get('writer_prompt'),
                            critic_description=self.agent_configs.get('critic_description'),
                            critic_prompt=self.agent_configs.get('critic_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                        
                    elif agent_type.lower() == 'editor':
                        agent_instance = EditorAgent(
                            max_steps=max_steps,
                            agent_description=self.agent_configs.get('editor_description'),
                            system_prompt=self.agent_configs.get('editor_prompt'),
                            fact_checker_description=self.agent_configs.get('fact_checker_description'),
                            fact_checker_prompt=self.agent_configs.get('fact_checker_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                        
                    elif agent_type.lower() == 'qaqc':
                        # Also remove original_row for QAQC agent if present
                        if 'original_row' in agent_context:
                            agent_context.pop('original_row')
                        
                        agent_instance = QAQCAgent(
                            max_steps=max_steps,
                            agent_description=self.agent_configs.get('qaqc_description'),
                            system_prompt=self.agent_configs.get('qaqc_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                    
                    elif agent_type.lower() == 'user_feedback':
                        # Get user email from environment or config
                        user_email = os.getenv("LOCAL_USER_EMAIL") or self.agent_configs.get('user_email')
                        report_frequency = self.agent_configs.get('report_frequency', 1)
                        
                        agent_instance = UserFeedbackAgent(
                            max_steps=max_steps,
                            user_email=user_email,
                            report_frequency=report_frequency,
                            agent_description=self.agent_configs.get('user_feedback_description'),
                            agent_prompt=self.agent_configs.get('user_feedback_prompt')
                        )
                        
                        # Log the feedback agent configuration
                        print(f"Initialized UserFeedbackAgent with email: {user_email}")
                        print(f"Using feedback agent system user: {FB_AGENT_USER}")
                        print(f"Report frequency: Every {report_frequency} iterations")
                        
                        self.agents[agent_type] = agent_instance
                    
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
                
                # Ensure the directory exists
                log_dir = os.path.dirname(self.state_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"Created directory: {log_dir}")
                
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
            
            # Use the load_file function to get the latest file from the previous agent
            try:
                file_content = load_file(agent_type=prev_agent_type)
                
                # Extract a title from the content
                title = "Previous Output"
                if isinstance(file_content, str) and file_content.startswith("File:"):
                    title_line = file_content.split("\n")[0]
                    title = title_line.replace("File:", "").strip().strip("'")
                
                # Add to prompt
                prompt += f"\n--- Latest Output from {prev_agent_type}: {title} ---\n"
                prompt += f"{file_content}\n"
                
                print(f"Added latest output from {prev_agent_type} to prompt")
            except Exception as e:
                print(f"Error loading file: {e}")
        
        # Add specific instructions based on agent type
        if agent_type == "researcher":
            # Add research parameters if available
            if hasattr(self, 'research_parameters'):
                prompt += "\n--- Research Parameters ---\n"
                for key, value in self.research_parameters.items():
                    if isinstance(value, list):
                        value_str = ", ".join(value)
                        prompt += f"{key}: {value_str}\n"
                    else:
                        prompt += f"{key}: {value}\n"
            
            # Add original row data if available
            if hasattr(self, 'original_row'):
                prompt += "\n--- Original Row Data ---\n"
                prompt += "\n".join([f"{key}={value}" for key, value in self.original_row.items()])
                prompt += "\n"
            
            prompt += "\nYour task is to research this topic thoroughly and provide comprehensive information with proper citations."
        elif agent_type == "writer":
            prompt += "\nYour task is to write engaging content based on the research provided."
        elif agent_type == "editor":
            prompt += "\nYour task is to edit and fact-check the content, ensuring accuracy while maintaining style and readability."
        elif agent_type == "qaqc":
            # For QAQC agent, we'll handle this differently in the run method
            # using the compare_outputs method directly
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
                    agent_instance = self.agents.get(agent_type)
                    
                    if not agent_instance:
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
                            
                            # Find the most recent non-user_feedback agent before QAQC
                            previous_agent_type = None
                            for i in range(qaqc_index - 1, -1, -1):
                                if self.agent_sequence[i] != "user_feedback":
                                    previous_agent_type = self.agent_sequence[i]
                                    break
                            
                            # If no suitable agent found before QAQC, look at the end of the sequence
                            if previous_agent_type is None:
                                for i in range(len(self.agent_sequence) - 1, qaqc_index, -1):
                                    if self.agent_sequence[i] != "user_feedback":
                                        previous_agent_type = self.agent_sequence[i]
                                        break
                            
                            # If still no suitable agent found, use the writer agent if available
                            if previous_agent_type is None and "writer" in self.agent_sequence:
                                previous_agent_type = "writer"
                            elif previous_agent_type is None and qaqc_index > 0:
                                # Fallback: use any non-user_feedback agent
                                for agent in self.agent_sequence:
                                    if agent != "user_feedback" and agent != "qaqc":
                                        previous_agent_type = agent
                                        break
                            
                            # Get the current and previous outputs from the selected agent
                            current_output = previous_results.get(f"{previous_agent_type}_{iteration}")
                            previous_output = previous_results.get(f"{previous_agent_type}_{iteration - 1}")
                            
                            # Print which agent we're comparing outputs from
                            print(f"QAQC is comparing outputs from: {previous_agent_type}")
                            
                            # Only proceed if we have outputs to compare
                            if current_output and previous_output:
                                # Run the comparison
                                result = agent_instance.compare_outputs([previous_output, current_output], query)
                                
                                # The compare_outputs method already handles selecting the best output
                                # and saving it to the appropriate file
                                
                                # Just log the result
                                print(f"QAQC result: {result[:200]}...")
                            else:
                                result = "Not enough outputs to compare"
                                print(result)
                        
                        # Special handling for UserFeedbackAgent
                        elif agent_type == "user_feedback":
                            # Create a state dictionary to pass to the UserFeedbackAgent
                            feedback_state = {
                                "iteration": iteration + 1,
                                "current_agent": agent_type,
                                "query": query,
                                "results": previous_results,
                                "agent_sequence": self.agent_sequence,
                                "feedback_agent_user": FB_AGENT_USER  # Add the feedback agent user to the state
                            }
                            
                            # Process feedback and update state
                            updated_state = agent_instance.process_feedback(feedback_state)
                            
                            # Get the result (this will be the report or feedback processing summary)
                            if agent_instance.should_report():
                                result = agent_instance.generate_report(feedback_state)
                                print(f"User feedback report sent via {FB_AGENT_USER}: {result[:200]}...")
                            else:
                                result = f"Checked for user feedback (reporting every {agent_instance.report_frequency} iterations)"
                                print(result)
                            
                            # Update the state with any user commands
                            if updated_state.get("user_commands"):
                                print(f"Received user commands: {updated_state['user_commands']}")
                                # Apply user commands to the state
                                for cmd, value in updated_state.get("user_commands", {}).items():
                                    if cmd == "frequency" and hasattr(agent_instance, "report_frequency"):
                                        agent_instance.report_frequency = value
                                        print(f"Updated report frequency to {value}")
                                    elif cmd == "feedback":
                                        print(f"User feedback: {value}")
                                        # Store the feedback in the state
                                        if "user_feedback" not in self.state:
                                            self.state["user_feedback"] = []
                                        self.state["user_feedback"].append({
                                            "iteration": iteration + 1,
                                            "time": time.time(),
                                            "feedback": value
                                        })
                        
                        else:
                            # Standard agent execution
                            if agent_type.lower() == 'researcher':
                                # Use run_query method for ResearcherAgent
                                result = agent_instance.run_query(formatted_prompt)
                            elif agent_type.lower() == 'writer':
                                # Use write_draft method for WriterAgent
                                result = agent_instance.write_draft(formatted_prompt)
                            elif agent_type.lower() == 'editor':
                                # Use edit_content method for EditorAgent
                                result = agent_instance.edit_content(formatted_prompt)
                            else:
                                # Other agents may implement __call__
                                result = agent_instance(formatted_prompt)
                        
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
    
    # Initialize the agent loop with default agent configs
    agent_configs = None
    if args.use_custom_prompts:
        # You could load custom configs here from a file if needed
        agent_configs = {}
    
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
        load_state=args.load_state,
        agent_configs=agent_configs,
        agent_contexts={}
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
    parser.add_argument("--state-file", type=str, default="logs/agent_loop_state.json", 
                        help="Path to a file for persisting state between runs")
    parser.add_argument("--load-state", action="store_true", 
                        help="Whether to load state from state_file if it exists (default: False)")
    
    return parser.parse_args()

if __name__ == "__main__":
    main() 