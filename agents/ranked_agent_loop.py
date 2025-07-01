#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from datetime import datetime
import uuid
import threading

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
from .ranker.agents import RankingAgent
from .ranker.tools import llm_judge

class RankedAgentLoop:
    """A class that manages a loop of agent calls with state management,
    artifact generation, and continuous ranking."""
    
    def __init__(
        self,
        agent_sequence: List[str],
        max_iterations: int = 5,
        max_steps_per_agent: Union[int, str] = 5,
        max_retries: int = 3,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        use_custom_prompts: bool = False,
        enable_telemetry: bool = False,
        agent_configs: Optional[Dict[str, Any]] = None,
        agent_contexts: Optional[Dict[str, Dict[str, Any]]] = None,
        ranking_llm_model_id: str = "gemini/gemini-1.5-flash",
        max_ranklist_size: int = 20,
        poll_interval: float = 5.0,
        primary_logical_artifact_id: str = "final_report",
        run_data_base_dir: str = "run_data",
        run_id: Optional[str] = None,
        load_run_state: bool = False
    ):
        """Initialize the ranked agent loop.
        
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
            agent_configs: Optional dictionary containing custom configuration for agents
            agent_contexts: Optional dictionary containing context data for specific agents
            ranking_llm_model_id: Model ID for the LLM judge in the ranking agent.
            max_ranklist_size: Maximum number of artifact IDs to retain in each ranklist.
            poll_interval: How often the ranking agent checks for new metadata (seconds).
            primary_logical_artifact_id: The main conceptual artifact being evolved.
            run_data_base_dir: Base directory for storing run-specific data.
            run_id: Specific ID for the run. If None, a new one is generated.
            load_run_state: If True and run_id is provided, attempts to load state for that run.
        """
        # Process agent_sequence first to remove 'ranker'
        # as it's handled separately and not part of the sequential execution loop.
        actual_agent_sequence_to_run = []
        ranker_was_in_sequence = False
        for agent_name in agent_sequence: # Iterate over the user-provided sequence
            if agent_name.lower() == "ranker":
                ranker_was_in_sequence = True
            else:
                actual_agent_sequence_to_run.append(agent_name)

        if ranker_was_in_sequence:
            print(
                "Warning: 'ranker' was found in the agent_sequence and has been removed. "
                "The RankingAgent is managed separately and is not part of the sequential agent execution loop. "
                "If you provided specific max_steps_per_agent for 'ranker', those steps will be ignored."
            )
        
        self.agent_sequence = actual_agent_sequence_to_run # This is now the cleaned sequence for the loop
        
        # Store basic configuration first
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.model_id = model_id
        self.model_info_path = model_info_path
        self.use_custom_prompts = use_custom_prompts
        self.enable_telemetry = enable_telemetry
        self.agent_configs = agent_configs or {}
        self.agent_contexts = agent_contexts or {}
        
        # Initialize run directory and paths first
        self.primary_logical_artifact_id = primary_logical_artifact_id
        self.run_data_base_dir = run_data_base_dir
        
        # Generate or set run_id early so it's available for all subsequent operations
        if run_id is None:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            print(f"Generated new run_id: {self.run_id}")
        else:
            self.run_id = run_id
            print(f"Using provided run_id: {self.run_id}")
            
        # Set up paths before attempting to load state
        self.run_dir = os.path.join(self.run_data_base_dir, self.run_id)
        self.artifacts_dir = os.path.join(self.run_dir, "artifacts")
        self.ranking_state_dir = os.path.join(self.run_dir, "ranking_state")
        self.metadata_path = os.path.join(self.run_dir, "metadata.jsonl")
        self.loop_state_file = os.path.join(self.run_dir, "agent_loop_state.json")

        # Create necessary directories
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.ranking_state_dir, exist_ok=True)
        
        # Parse max_steps_per_agent
        if isinstance(max_steps_per_agent, str) and "," in max_steps_per_agent:
            # Parse comma-separated list of max steps
            steps_list = [int(steps.strip()) for steps in max_steps_per_agent.split(",")]
            self.max_steps_per_agent_dict = {}
            
            # Map steps to agents in the cleaned self.agent_sequence
            for i, agent_type in enumerate(self.agent_sequence):
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
            self.max_steps_per_agent_dict = {agent_type: self.max_steps_per_agent for agent_type in self.agent_sequence}
        
        # Initialize state
        self.state = {
            "run_id": self.run_id,
            "status": "initialized",
            "current_iteration": 0,
            "current_agent_index": 0,
            "results": {},
            "cycle_input_artifact_ids": {},
            "error": None,
            "last_updated": time.time(),
            "query": None,
            "ranking_started": False,
            "current_best_artifact_id": None,  # Track the best artifact before ranking starts
            "ranking_status": {
                "last_update": None,
                "ranklist": [],
                "comparisons": [],  # List of comparison results with rationale
                "total_artifacts_processed": 0
            }
        }
        
        # Track if ranking has been started in a property too
        self.ranking_started = False
        
        # Initialize the file manager
        self.file_manager = FileManager()
        
        # Load state from file if provided and load_run_state is True
        if load_run_state and self.run_id:
            if os.path.exists(self.loop_state_file):
                try:
                    with open(self.loop_state_file, 'r') as f:
                        loaded_state = json.load(f)
                        # Selectively update, don't overwrite run_id or paths
                        self.state["status"] = loaded_state.get("status", "initialized")
                        self.state["current_iteration"] = loaded_state.get("current_iteration", 0)
                        self.state["current_agent_index"] = loaded_state.get("current_agent_index", 0)
                        self.state["results"] = loaded_state.get("results", {})
                        self.state["cycle_input_artifact_ids"] = loaded_state.get("cycle_input_artifact_ids", {})
                        self.state["query"] = loaded_state.get("query")
                        self.state["last_updated"] = loaded_state.get("last_updated", time.time())
                        self.state["ranking_started"] = loaded_state.get("ranking_started", False)
                        self.state["current_best_artifact_id"] = loaded_state.get("current_best_artifact_id")
                        self.ranking_started = self.state["ranking_started"]
                        
                        # If ranking was previously started and we're reloading, start it again
                        if self.ranking_started:
                            print(f"Resuming ranking agent for run {self.run_id}")
                            self.ranking_agent.start_background_ranking()
                            
                        print(f"Loaded state from {self.loop_state_file} for run {self.run_id}")
                except Exception as e:
                    print(f"Error loading state from {self.loop_state_file} for run {self.run_id}: {e}. Starting fresh for this run.")
            else:
                print(f"State file {self.loop_state_file} not found for run {self.run_id}. Starting fresh for this run.")
        
        # Set up environment
        self._setup_environment()
        
        # Initialize agent configs with provided configs or defaults
        if use_custom_prompts:
            default_configs = self._get_default_agent_configs()
            # Update with any provided configs
            if agent_configs:
                default_configs.update(agent_configs)
            self.agent_configs = default_configs
            
        # Create LLM models BEFORE initializing agents
        # Create main model for agents
        self.main_llm_model = RateLimitedLiteLLMModel(
            model_id=self.model_id,
            model_info_path=self.model_info_path,
            base_wait_time=2.0,
            max_retries=self.max_retries
        )
        
        # Store ranking configuration
        self.ranking_llm_model_id = ranking_llm_model_id
        self.max_ranklist_size = max_ranklist_size
        self.poll_interval = poll_interval
        
        # Create LLM model for ranking
        self.ranking_llm_model = RateLimitedLiteLLMModel(
            model_id=self.ranking_llm_model_id,
            model_info_path=self.model_info_path,
            base_wait_time=2.0,
            max_retries=self.max_retries
        )
        
        # Create a shared lock for metadata access
        self.metadata_lock = threading.Lock()
        
        # Initialize agents now that all models are created
        self.agents = {}
        self._initialize_agents()
        
        # Initialize the ranking agent after agents are initialized
        self.ranking_agent = RankingAgent(
            model=self.ranking_llm_model,
            max_ranklist_size=self.max_ranklist_size,
            poll_interval=self.poll_interval,
            run_data_base_dir=self.run_data_base_dir,
            run_id=self.run_id,
            logical_artifact_id=self.primary_logical_artifact_id,
            parent_loop=self  # Pass reference to the parent loop for status updates
        )
        
        # Don't start the background ranking thread until we have 2 artifacts
        self.ranking_started = False
        
        # Initialize empty ranklist file to prevent file not found errors
        ranklist_path = os.path.join(self.ranking_state_dir, f"{self.primary_logical_artifact_id}.ranklist.json")
        if not os.path.exists(ranklist_path):
            print(f"Proactively initializing empty ranklist file for '{self.primary_logical_artifact_id}'")
            with open(ranklist_path, 'w') as f:
                json.dump([], f)
        
        print(f"RankingAgent for '{self.primary_logical_artifact_id}' initialized for run '{self.run_id}'. Will start ranking after 2 artifacts are created.")
        
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
                agent_name="ranked_agent_loop", 
                agent_type=f"ranked_loop_with_{','.join(self.agent_sequence)}"
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
            
            "user_email": os.getenv("REMOTE_USER_EMAIL", "example@example.com"),  # External user email for feedback
            "report_frequency": 1,
            # Description for UserFeedbackAgent's internal report-generating LLM
            "user_feedback_agent_description": "Generates concise progress reports based on provided state.",
            # Prompt for UserFeedbackAgent's internal report-generating LLM
            "user_feedback_agent_prompt": "You are an assistant that writes brief, informative progress reports based on the current state of an automated loop of agents working on a task. Focus on key achievements, changes, problems, errors, etc. that occurred during this iteration. Do not include anything else but the progress report."
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
                        
                        # Extract additional_tools if provided
                        additional_tools = agent_context.get('additional_tools')
                        
                        agent_instance = ResearcherAgent(
                            model=self.main_llm_model,
                            max_steps=max_steps,
                            researcher_description=self.agent_configs.get('researcher_description'),
                            researcher_prompt=self.agent_configs.get('researcher_prompt'),
                            additional_tools=additional_tools
                        )
                        self.agents[agent_type] = agent_instance
                        
                    elif agent_type.lower() == 'writer':
                        agent_instance = WriterAgent(
                            model=self.main_llm_model,
                            max_steps=max_steps,
                            agent_description=self.agent_configs.get('writer_description'),
                            system_prompt=self.agent_configs.get('writer_prompt'),
                            critic_description=self.agent_configs.get('critic_description'),
                            critic_prompt=self.agent_configs.get('critic_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                        
                    elif agent_type.lower() == 'editor':
                        agent_instance = EditorAgent(
                            model=self.main_llm_model,
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
                            model=self.main_llm_model,
                            max_steps=max_steps,
                            agent_description=self.agent_configs.get('qaqc_description'),
                            system_prompt=self.agent_configs.get('qaqc_prompt')
                        )
                        self.agents[agent_type] = agent_instance
                    
                    elif agent_type.lower() == 'user_feedback':
                        # Get user email from environment or config
                        remote_email = os.getenv("REMOTE_USER_EMAIL") or self.agent_configs.get('user_email')
                        report_frequency = self.agent_configs.get('report_frequency', 1)
                        
                        agent_instance = UserFeedbackAgent(
                            model=self.main_llm_model,
                            max_steps=max_steps, # Max steps for the internal report_generator_agent
                            user_email=remote_email,
                            report_frequency=report_frequency,
                            agent_description=self.agent_configs.get('user_feedback_agent_description'), # For the report_generator_agent
                            agent_prompt=self.agent_configs.get('user_feedback_agent_prompt') # For the report_generator_agent
                        )
                        
                        # Log the feedback agent configuration
                        print(f"Initialized UserFeedbackAgent:")
                        print(f"- External email (sending to/receiving from): {agent_instance.remote_email or 'Not configured'}")
                        print(f"- Local email (sending from): {agent_instance.local_email or 'Not configured'}")
                        # FB_AGENT_USER is an internal detail of the tools used by UserFeedbackAgent
                        print(f"- Report frequency (initial): Every {report_frequency} iterations (can be changed by email command)")
                        
                        self.agents[agent_type] = agent_instance
                    
                    print(f"Initialized agent: {agent_type}")
                except Exception as e:
                    print(f"Error initializing agent {agent_type}: {e}")
                    self.state["error"] = f"Error initializing agent {agent_type}: {str(e)}"
                    self.state["status"] = "error"
                    self._save_loop_state()
                    return self.state
    
    def _save_loop_state(self):
        """Saves the current state of the agent loop to its run-specific state file."""
        if self.loop_state_file:
            try:
                self.state["last_updated"] = time.time()
                # Ensure the directory exists (should be created in __init__)
                os.makedirs(os.path.dirname(self.loop_state_file), exist_ok=True)
                
                with open(self.loop_state_file, 'w') as f:
                    json.dump(self.state, f, indent=2)
                print(f"Saved loop state for run {self.run_id} to {self.loop_state_file}")
            except Exception as e:
                print(f"Error saving loop state for run {self.run_id} to {self.loop_state_file}: {e}")
    
    def _format_prompt_for_agent(self, agent_type: str, query: str, current_artifact_content: Optional[str], iteration: int) -> str:
        """
        Formats a prompt for a specific agent type.
        Now includes current_artifact_content for the primary logical artifact.
        """
        prompt = f"Run ID: {self.run_id}, Iteration: {iteration}\n"
        prompt += f"Original query/goal for '{self.primary_logical_artifact_id}': {query}\n\n"

        if current_artifact_content:
            prompt += f"--- Current Best Version of '{self.primary_logical_artifact_id}' (Input for this step) ---\n"
            prompt += f"{current_artifact_content}\n\n"
        else:
            prompt += f"--- No existing version of '{self.primary_logical_artifact_id}' provided. You are creating the first version or working from the query alone. ---\n\n"

        # Add context from other (non-primary) artifacts if necessary, or previous agent results *within the cycle*
        # This part needs careful design based on how agents interact beyond the primary artifact.
        # For now, focusing on the primary artifact evolution.
        # `previous_results` from original loop might be used for intermediate data passing within a cycle.
        # For example, if writer passes to editor *within the same iteration N*, that's not via ranking.
        
        # Example: if self.state['results'] holds *intra-cycle* data:
        # if self.state['results'] and agent_type.lower() != self.agent_sequence[0].lower(): # if not first agent
        #    prompt += "Intermediate results from previous agent in this cycle:\n"
        #    # Logic to find the immediate previous agent's output from self.state['results']
        #    # This is complex and depends on how `self.state['results']` is populated.
        #    # Let's assume for now agents mainly operate on the primary ranked artifact.

        # Add specific instructions based on agent type (similar to original)
        if agent_type.lower() == "researcher":
             prompt += f"\nYour task is to research information relevant to the primary goal: '{query}'. The current best version of '{self.primary_logical_artifact_id}' is provided above. Use your research to help improve it or gather supplementary data. Focus on contributing to the evolution of '{self.primary_logical_artifact_id}'."
        elif agent_type.lower() == "writer":
            prompt += f"\nYour task is to write or revise the content for '{self.primary_logical_artifact_id}' based on the provided version (if any) and the overall goal: '{query}'. Focus on producing a new, improved version of '{self.primary_logical_artifact_id}'."
        elif agent_type.lower() == "editor":
            prompt += f"\nYour task is to edit and fact-check the provided version of '{self.primary_logical_artifact_id}', ensuring accuracy and improving its quality in line with the goal: '{query}'. Produce a new, edited version of '{self.primary_logical_artifact_id}'."
        # Add other agent-specific instructions here.
        else:
            prompt += f"\nYour task is to process the provided version of '{self.primary_logical_artifact_id}' according to your role as a '{agent_type}' agent, contributing to the overall goal: '{query}'. Produce a new version of '{self.primary_logical_artifact_id}'."
        
        return prompt
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent loop for a given query.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary containing the final results and state information
        """
        print(f"Starting ranked agent loop for run '{self.run_id}' with query: {query}")
        print(f"Agent sequence: {' -> '.join(self.agent_sequence)}")
        print(f"Primary logical artifact: '{self.primary_logical_artifact_id}'")

        # If query is not loaded from state, set it
        if self.state.get("query") is None:
            self.state["query"] = query
        elif self.state["query"] != query and not self.state.get("current_iteration",0) > 0: # if query changes for a fresh run
            print(f"Query changed from '{self.state['query']}' to '{query}'. Updating.")
            self.state["query"] = query

        if self.state["status"] == "completed":
            print(f"Run {self.run_id} was already completed. To re-run, use a new run_id or delete existing run_data.")
            return self.state # For now, don't auto-reset

        self.state["status"] = "running"
        
        try:
            # Loop through iterations
            while self.state["current_iteration"] < self.max_iterations:
                iteration_num = self.state["current_iteration"]
                print(f"\n--- Run {self.run_id} - Iteration {iteration_num + 1}/{self.max_iterations} ---")

                # Check if we should start the ranking agent at the beginning of the cycle
                if not self.ranking_started:
                    artifact_count = self._count_artifacts_for_logical_id(self.primary_logical_artifact_id)
                    if artifact_count >= 2:
                        print(f"Starting ranking agent after detecting {artifact_count} artifacts at start of cycle {iteration_num + 1}")
                        self.ranking_agent.start_background_ranking()
                        self.ranking_started = True
                        self.state["ranking_started"] = True
                        print(f"RankingAgent for '{self.primary_logical_artifact_id}' started for run '{self.run_id}'.")

                # 1. Determine input artifact for this cycle (Start of Cycle Logic)
                current_best_artifact_id = None
                current_artifact_content_for_cycle = None
                
                # Check if we're in early iterations (before ranking starts) or if ranking has started
                if iteration_num < 2 or not self.ranking_started:
                    # Early iterations or ranking not started yet - use previous artifact if available
                    print(f"Iteration {iteration_num + 1} (Pre-ranking) - Using most recent artifact as input")
                    
                    # For the first iteration, current_best_artifact_id will remain None
                    if iteration_num > 0:
                        # First check if we have a current best artifact in the state
                        if "current_best_artifact_id" in self.state and self.state["current_best_artifact_id"]:
                            current_best_artifact_id = self.state["current_best_artifact_id"]
                            print(f"Cycle {iteration_num + 1} input: Using current best artifact ID '{current_best_artifact_id}' from state.")
                        else:
                            # Fallback: For subsequent iterations before ranking starts, use the last artifact from the previous iteration
                            # Get the last generated artifact ID for this logical artifact
                            previous_iter_artifacts = []
                            if os.path.exists(self.metadata_path):
                                with open(self.metadata_path, 'r') as f:
                                    for line in f:
                                        try:
                                            entry = json.loads(line.strip())
                                            if (entry.get("logical_artifact_id") == self.primary_logical_artifact_id and
                                                entry.get("iteration") == iteration_num):
                                                previous_iter_artifacts.append(entry)
                                        except json.JSONDecodeError:
                                            continue
                                            
                            # Sort by timestamp if multiple entries exist
                            if previous_iter_artifacts:
                                previous_iter_artifacts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                                current_best_artifact_id = previous_iter_artifacts[0].get("artifact_id")
                                print(f"Cycle {iteration_num + 1} input: Using most recent artifact ID '{current_best_artifact_id}' from previous iteration.")
                            else:
                                print(f"No artifacts found for '{self.primary_logical_artifact_id}' from iteration {iteration_num}.")
                else:
                    # Ranking has started - use the top-ranked artifact
                    ranklist_path_for_primary = os.path.join(self.ranking_state_dir, f"{self.primary_logical_artifact_id}.ranklist.json")
                    
                    ranklist_exists = os.path.exists(ranklist_path_for_primary)
                    ranklist_has_entries = False
                    
                    if ranklist_exists:
                        try:
                            with open(ranklist_path_for_primary, 'r') as f:
                                content = f.read().strip()
                                if content: # Check if file is not empty
                                    ranklist = json.loads(content)
                                    if ranklist: # Check if list is not empty
                                        ranklist_has_entries = True
                                        current_best_artifact_id = ranklist[0]
                                        print(f"Cycle {iteration_num + 1} input: Top artifact ID '{current_best_artifact_id}' from '{ranklist_path_for_primary}'.")
                                    else:
                                        print(f"Ranklist file '{ranklist_path_for_primary}' exists but contains empty list. Using most recent artifact instead.")
                                else:
                                    print(f"Ranklist file '{ranklist_path_for_primary}' exists but is empty. Using most recent artifact instead.")
                        except Exception as e:
                            print(f"Error reading ranklist {ranklist_path_for_primary}: {e}. Using most recent artifact instead.")
                    else:
                        # If ranklist doesn't exist, create an empty one to prevent future errors
                        print(f"Ranklist file '{ranklist_path_for_primary}' not found. Creating empty ranklist and using most recent artifact.")
                        os.makedirs(os.path.dirname(ranklist_path_for_primary), exist_ok=True)
                        with open(ranklist_path_for_primary, 'w') as f:
                            json.dump([], f)
                    
                    # If we couldn't get a ranked artifact, fall back to most recent artifact
                    if current_best_artifact_id is None:
                        # Similar logic as above for finding most recent artifact
                        all_artifacts = []
                        if os.path.exists(self.metadata_path):
                            with open(self.metadata_path, 'r') as f:
                                for line in f:
                                    try:
                                        entry = json.loads(line.strip())
                                        if entry.get("logical_artifact_id") == self.primary_logical_artifact_id:
                                            all_artifacts.append(entry)
                                    except json.JSONDecodeError:
                                        continue
                                        
                        # Sort by timestamp if multiple entries exist
                        if all_artifacts:
                            all_artifacts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                            current_best_artifact_id = all_artifacts[0].get("artifact_id")
                            print(f"Cycle {iteration_num + 1} fallback: Using most recent artifact ID '{current_best_artifact_id}'.")

                self.state['cycle_input_artifact_ids'][self.primary_logical_artifact_id] = current_best_artifact_id
                
                # Load content of the current best artifact
                current_artifact_content_for_cycle = self._load_artifact_content(current_best_artifact_id)
                
                # Store this to be passed through the cycle if agents modify it sequentially
                # This `current_cycle_artifact_content` will be updated by agents if they are chained
                # within this iteration to operate on the output of the previous one.
                next_agent_input_content = current_artifact_content_for_cycle

                # Process each agent in the sequence for this iteration
                agent_idx = self.state.get("current_agent_index", 0)
                while agent_idx < len(self.agent_sequence):
                    self.state["current_agent_index"] = agent_idx
                    agent_type = self.agent_sequence[agent_idx]
                    agent_instance = self.agents.get(agent_type)

                    if not agent_instance:
                        # This should ideally be caught in _initialize_agents
                        error_msg = f"Agent {agent_type} not initialized for run {self.run_id}"
                        print(f"Error: {error_msg}")
                        self.state["error"] = error_msg; self.state["status"] = "error"
                        self._save_loop_state(); return self.state

                    print(f"\nRunning agent: {agent_type} (Iteration {iteration_num + 1})")

                    # Format prompt using the latest version of the artifact for this cycle/agent step
                    # If agent A produces output, and agent B uses it IN THE SAME CYCLE,
                    # `next_agent_input_content` should be that output.
                    formatted_prompt = self._format_prompt_for_agent(agent_type, self.state["query"], next_agent_input_content, iteration_num + 1)
                    
                    agent_result_content = None
                    try:
                        start_time = time.time()
                        # --- Agent Execution Logic ---
                        # Agents are expected to return the content of the *new* artifact they created.
                        if agent_type.lower() == 'researcher':
                            agent_result_content = agent_instance.run_query(formatted_prompt)
                        elif agent_type.lower() == 'writer':
                            agent_result_content = agent_instance.write_draft(formatted_prompt)
                        elif agent_type.lower() == 'editor':
                            agent_result_content = agent_instance.edit_content(formatted_prompt)
                        elif agent_type.lower() == 'user_feedback':
                            # UserFeedbackAgent now handles its email cycle internally.
                            # It takes the loop's current state for context to generate reports
                            # and can update the loop's state (e.g., 'paused').
                            feedback_state_input = {
                                "iteration": iteration_num + 1, 
                                "current_agent_loop_agent": agent_type, 
                                "query": self.state["query"],
                                "loop_results": {k: v for k, v in self.state["results"].items() if f"_iter{iteration_num}" in k or not "_iter" in k}, # Pass current iter results or general results
                                "agent_sequence": self.agent_sequence,
                                "paused": self.state.get("paused", False)
                            }
                            
                            # process_feedback now returns the updated state, including any parsed commands
                            # and potentially a 'paused' status.
                            updated_feedback_state = agent_instance.process_feedback(feedback_state_input)
                            
                            # Update the main loop's state based on feedback (e.g., pause)
                            if "paused" in updated_feedback_state:
                                self.state["paused"] = updated_feedback_state["paused"]
                                print(f"Loop pause state updated by UserFeedbackAgent to: {self.state['paused']}")

                            # Log processed commands
                            if "user_commands" in updated_feedback_state and updated_feedback_state["user_commands"]:
                                if "user_feedback_commands_log" not in self.state:
                                    self.state["user_feedback_commands_log"] = []
                                self.state["user_feedback_commands_log"].append({
                                    "iteration": iteration_num + 1,
                                    "time": time.time(),
                                    "commands": updated_feedback_state["user_commands"]
                                })
                                print(f"User commands processed by UserFeedbackAgent: {updated_feedback_state['user_commands']}")

                            # The "result" of the UserFeedbackAgent's turn is a summary of its actions.
                            agent_result_content = f"UserFeedbackAgent cycle {iteration_num + 1}: Checked mail. "
                            if agent_instance.should_report() and not self.state.get("paused", False) and agent_instance.remote_email:
                                agent_result_content += "Attempted to send report. "
                            if updated_feedback_state.get("user_commands"):
                                agent_result_content += f"Processed commands: {updated_feedback_state['user_commands']}. "
                            if self.state.get("paused", False):
                                agent_result_content += "Loop is PAUSED. "
                            
                            print(agent_result_content) # Print the summary
                            # The UserFeedbackAgent does not produce a new version of the primary_logical_artifact_id.
                            # Its output (summary string) will be stored in self.state["results"] by the generic logic below.
                        else:
                            # Generic call for other agent types
                            agent_result_content = agent_instance(formatted_prompt) 
                        
                        end_time = time.time()
                        print(f"Agent {agent_type} completed in {end_time - start_time:.2f}s.")

                        if agent_result_content is not None:
                            # If agent is NOT user_feedback, assume it produced a new version of the primary artifact
                            if agent_type.lower() != 'user_feedback':
                                new_artifact_id = self._save_artifact(
                                    content=agent_result_content,
                                    logical_artifact_id=self.primary_logical_artifact_id,
                                    agent_type=agent_type,
                                    iteration=iteration_num + 1 # Use 1-based for filename
                                )
                                print(f"Agent {agent_type} produced new artifact ID: {new_artifact_id} for '{self.primary_logical_artifact_id}'.")
                                
                                # This new content becomes the input for the next agent *in this cycle*
                                next_agent_input_content = agent_result_content 
                                
                                # Store this iteration's specific output under a unique key if needed for intra-cycle analysis
                                self.state["results"][f"{agent_type}_iter{iteration_num}_artifact_id"] = new_artifact_id
                                
                                # For early iterations (before ranking starts), use this artifact as the current best
                                # Store in a special key in the state to track the "current best" before ranking starts
                                if not self.ranking_started:
                                    self.state["current_best_artifact_id"] = new_artifact_id
                                    print(f"Updated current best artifact to: {new_artifact_id} (pre-ranking mode)")
                                    
                                    # Check if we now have 2 artifacts and can start ranking
                                    artifact_count = self._count_artifacts_for_logical_id(self.primary_logical_artifact_id)
                                    if artifact_count >= 2:
                                        print(f"Starting ranking agent after artifact {new_artifact_id} created (total: {artifact_count} artifacts)")
                                        self.ranking_agent.start_background_ranking()
                                        self.ranking_started = True
                                        self.state["ranking_started"] = True
                                        print(f"RankingAgent for '{self.primary_logical_artifact_id}' started for run '{self.run_id}'.")
                                    
                                # If this is the last agent in the sequence for this iteration, make a note
                                if agent_idx == len(self.agent_sequence) - 1:
                                    print(f"Completed iteration {iteration_num + 1} with final artifact: {new_artifact_id}")
                            else:
                                # For user_feedback or other non-artifact-producing agents
                                print(f"Agent {agent_type} result: {agent_result_content[:200]}...")
                                self.state["results"][f"{agent_type}_iter{iteration_num}_status"] = agent_result_content

                        # Store general result (e.g. summary, status, or the content itself)
                        # This 'results' is more for logging/debugging the cycle's outputs
                        # The ranked artifact is the source of truth for the primary logical artifact.
                        result_summary_key = f"{agent_type}_{iteration_num}"
                        self.state["results"][result_summary_key] = agent_result_content[:500] + "..." if agent_result_content and len(agent_result_content) > 500 else agent_result_content
                        
                    except Exception as e:
                        error_msg = f"Error running agent {agent_type} in iter {iteration_num + 1} for run {self.run_id}: {e}"
                        print(f"Error: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        self.state["error"] = error_msg; self.state["status"] = "error"
                        self._save_loop_state(); return self.state
                    
                    agent_idx += 1
                    self.state["current_agent_index"] = agent_idx # Update before saving
                    self._save_loop_state() # Save state after each agent step within an iteration

                # Completed one full iteration through agent sequence
                self.state["current_iteration"] += 1
                self.state["current_agent_index"] = 0 # Reset for next iteration
                
                # Start the ranking agent as soon as there are at least 2 artifacts
                if not self.ranking_started:
                    artifact_count = self._count_artifacts_for_logical_id(self.primary_logical_artifact_id)
                    if artifact_count >= 2:
                        print(f"Starting ranking agent after detecting {artifact_count} artifacts")
                        self.ranking_agent.start_background_ranking()
                        self.ranking_started = True
                        self.state["ranking_started"] = True
                        print(f"RankingAgent for '{self.primary_logical_artifact_id}' started for run '{self.run_id}'.")
                
                self._save_loop_state() # Save state after each full iteration

            # Loop completed successfully
            self.state["status"] = "completed"
            print(f"Ranked agent loop for run '{self.run_id}' completed all {self.max_iterations} iterations.")
            
        except Exception as e:
            error_msg = f"Unexpected error in ranked agent loop for run {self.run_id}: {e}"
            print(f"Error: {error_msg}")
            import traceback
            traceback.print_exc()
            self.state["error"] = error_msg; self.state["status"] = "error"
        finally:
            self._save_loop_state()
            return self.state

    def close(self):
        """Shuts down the ranking agent and performs cleanup."""
        if hasattr(self, 'run_id'):
            print(f"Closing RankedAgentLoop for run {self.run_id}...")
        else:
            print(f"Closing RankedAgentLoop...")
        
        if hasattr(self, 'ranking_agent') and self.ranking_agent:
            try:
                # If ranking was started, give it time to process final artifacts
                if self.ranking_started:
                    # Get the count of artifacts in metadata
                    current_count = self._count_artifacts_for_logical_id(self.primary_logical_artifact_id)
                    
                    # Read current ranklist
                    ranklist_path = os.path.join(self.ranking_state_dir, f"{self.primary_logical_artifact_id}.ranklist.json")
                    ranked_count = 0
                    if os.path.exists(ranklist_path):
                        try:
                            with open(ranklist_path, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    ranklist = json.loads(content)
                                    ranked_count = len(ranklist)
                        except Exception as e:
                            print(f"Error reading final ranklist: {e}")
                    
                    # If not all artifacts ranked, give the ranker time to catch up
                    if ranked_count < current_count:
                        print(f"Waiting for ranking agent to process final artifacts ({ranked_count}/{current_count} ranked)...")
                        # Wait up to 5 poll intervals for the ranker to catch up
                        max_wait = 5
                        for i in range(max_wait):
                            time.sleep(self.ranking_agent.poll_interval)
                            # Check if ranking caught up
                            if os.path.exists(ranklist_path):
                                try:
                                    with open(ranklist_path, 'r') as f:
                                        content = f.read().strip()
                                        if content:
                                            ranklist = json.loads(content)
                                            if len(ranklist) >= current_count:
                                                print(f"Ranking caught up after {i+1} polls.")
                                                break
                                except Exception:
                                    pass
                            if i == max_wait - 1:
                                print(f"Not all artifacts ranked after waiting. Proceeding with shutdown.")
                
                # Now stop the ranking agent
                self.ranking_agent.stop_background_ranking()
            except Exception as e:
                print(f"Error stopping ranking agent: {e}")
        
        print("RankedAgentLoop closed.")

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected."""
        try:
            self.close()
        except Exception as e:
            # Silent cleanup - print only if in debug mode
            print(f"Error during cleanup: {e}")

    def _get_artifact_full_path(self, artifact_id: str) -> Optional[str]:
        """Looks up artifact's relative_path in metadata.jsonl and returns full path."""
        # This needs to read metadata.jsonl carefully with locking
        # For simplicity in this step, assume RankingAgent handles metadata reads primarily
        # This function would be more robust by reading metadata.jsonl itself
        # For now, we'll construct a potential path and check. A more robust way is needed.
        
        # TEMPORARY: This is a placeholder. Robust metadata lookup is needed.
        # This function would ideally search self.metadata_path
        # For now, let's assume a convention or that RankingAgent has a way to provide this.
        # This method is CRITICAL and needs a proper implementation that reads metadata.jsonl.
        
        # Correct implementation requires reading metadata.jsonl
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file {self.metadata_path} not found. Cannot get artifact path for {artifact_id}")
            return None

        with self.metadata_lock: # Ensure thread-safe read of metadata
            with open(self.metadata_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("artifact_id") == artifact_id and "relative_path" in entry:
                            return os.path.join(self.run_dir, entry["relative_path"])
                    except json.JSONDecodeError:
                        continue # Skip malformed lines
        print(f"Artifact ID {artifact_id} not found or missing 'relative_path' in {self.metadata_path}")
        return None

    def _load_artifact_content(self, artifact_id: Optional[str]) -> Optional[str]:
        """Loads artifact content given its ID."""
        if artifact_id is None:
            print("Attempted to load artifact with None ID.")
            return None
            
        full_path = self._get_artifact_full_path(artifact_id)
        if full_path and os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading artifact file {full_path}: {e}")
                return None
        else:
            if full_path:
                 print(f"Artifact file not found at resolved path: {full_path} for ID {artifact_id}")
            else:
                 print(f"Could not resolve path for artifact ID: {artifact_id}")
            return None

    def _save_artifact(self, content: str, logical_artifact_id: str, agent_type: str, iteration: int) -> str:
        """
        Saves artifact content to a unique file, logs metadata.
        Returns the unique artifact_id.
        """
        new_artifact_id = uuid.uuid4().hex
        timestamp = datetime.now().isoformat()
        
        # Define a short logical ID for filename, avoid special chars
        logical_id_short = "".join(c if c.isalnum() else "_" for c in logical_artifact_id)[:20]
        artifact_id_short = new_artifact_id[:8]

        # Recommended Naming Convention: iter<N>_<agent_type>_<logical_id_short>_<artifact_id_short>.<ext>
        # For simplicity, using .txt, can be made more dynamic
        ext = "md" if "report" in logical_artifact_id or "draft" in logical_artifact_id else "txt"
        filename = f"iter{iteration}_{agent_type}_{logical_id_short}_{artifact_id_short}.{ext}"
        relative_path = os.path.join("artifacts", filename) # Relative to run_dir
        full_path = os.path.join(self.run_dir, relative_path)

        try:
            print(f"Saving artifact to {full_path}")
            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved new artifact version: {full_path} (ID: {new_artifact_id})")
        except Exception as e:
            print(f"Error saving artifact {full_path}: {e}")
            import traceback
            traceback.print_exc()
            # Decide how to handle this error - re-raise?
            raise

        metadata_entry = {
            "artifact_id": new_artifact_id,
            "logical_artifact_id": logical_artifact_id,
            "relative_path": relative_path,
            "agent_type": agent_type,
            "timestamp": timestamp,
            "word_count": len(content.split()), # Example simple metadata
            "iteration": iteration # Track which iteration produced it
        }

        try:
            print(f"Appending metadata for artifact {new_artifact_id} to {self.metadata_path}")
            with self.metadata_lock:
                # Ensure the directory for metadata exists
                os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
                
                with open(self.metadata_path, 'a') as f: # Append mode
                    json.dump(metadata_entry, f)
                    f.write('\n') # Newline for JSONL
            print(f"Appended metadata for artifact {new_artifact_id} to {self.metadata_path}")
        except Exception as e:
            print(f"Error appending metadata for artifact {new_artifact_id}: {e}")
            import traceback
            traceback.print_exc()
            # Critical error, might need to handle cleanup or stop
            raise
            
        return new_artifact_id

    def _count_artifacts_for_logical_id(self, logical_artifact_id: str) -> int:
        """Count the number of artifacts for a given logical artifact ID.
        
        Args:
            logical_artifact_id: The logical artifact ID to count artifacts for
            
        Returns:
            The number of artifacts
        """
        count = 0
        if os.path.exists(self.metadata_path):
            try:
                with self.metadata_lock:  # Use lock for thread safety
                    with open(self.metadata_path, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("logical_artifact_id") == logical_artifact_id:
                                    count += 1
                            except json.JSONDecodeError:
                                continue  # Skip invalid lines
            except Exception as e:
                print(f"Error counting artifacts for {logical_artifact_id}: {e}")
        
        return count

    def update_ranking_status(self, new_ranklist=None, new_comparison=None):
        """Updates the ranking status in the agent loop state.
        
        Args:
            new_ranklist: Optional updated ranklist
            new_comparison: Optional dictionary with comparison details including rationale
                Example: {"artifact_a": "id1", "artifact_b": "id2", "winner": "A", "rationale": "..."}
        """
        with self.metadata_lock:  # Use lock for thread safety
            # Update ranking status
            self.state["ranking_status"]["last_update"] = time.time()
            
            if new_ranklist is not None:
                self.state["ranking_status"]["ranklist"] = new_ranklist
            
            if new_comparison is not None:
                # Add the new comparison to the comparisons list
                self.state["ranking_status"]["comparisons"].append(new_comparison)
                # Increment the total artifacts processed counter
                self.state["ranking_status"]["total_artifacts_processed"] += 1
                
            # Save the updated state
            self._save_loop_state()

def main():
    """Main entry point for the agent loop."""
    args = parse_args()
    
    print("=== Ranked Agent Loop ===")
    print(f"Query: {args.query}")
    print(f"Agent Sequence: {args.agent_sequence}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Max Steps Per Agent: {args.max_steps_per_agent}")
    print(f"Max Retries: {args.max_retries}")
    print(f"Model ID: {args.model_id}")
    print(f"Model Info Path: {args.model_info_path}")
    print(f"Use Custom Prompts: {args.use_custom_prompts}")
    print(f"Enable Telemetry: {args.enable_telemetry}")
    print(f"Run ID: {args.run_id}")
    print(f"Load Run State: {args.load_run_state}")
    print()
    
    # Parse agent sequence
    agent_sequence = [agent_type.strip() for agent_type in args.agent_sequence.split(",")]
    
    # Initialize the ranked agent loop
    ranked_loop = RankedAgentLoop(
        agent_sequence=agent_sequence,
        max_iterations=args.max_iterations,
        max_steps_per_agent=args.max_steps_per_agent,
        max_retries=args.max_retries,
        model_id=args.model_id,
        model_info_path=args.model_info_path,
        use_custom_prompts=args.use_custom_prompts,
        enable_telemetry=args.enable_telemetry,
        ranking_llm_model_id=args.ranking_llm_model_id,
        max_ranklist_size=args.max_ranklist_size,
        poll_interval=args.poll_interval,
        primary_logical_artifact_id=args.primary_logical_artifact_id,
        run_data_base_dir=args.run_data_base_dir,
        run_id=args.run_id,
        load_run_state=args.load_run_state
    )
    
    final_state = None
    try:
        final_state = ranked_loop.run(args.query)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down gracefully...")
    finally:
        ranked_loop.close() # Ensure ranking agent is shut down

    if final_state:
        if final_state["status"] == "completed":
            print("\n=== Final Loop State ===")
            # Print key details from final_state
            print(f"Run ID: {final_state.get('run_id')}")
            print(f"Status: {final_state.get('status')}")
            print(f"Iterations Completed: {final_state.get('current_iteration')}")
            
            # Try to display the best artifact
            ranklist_path = os.path.join(ranked_loop.run_dir, "ranking_state", f"{ranked_loop.primary_logical_artifact_id}.ranklist.json")
            if os.path.exists(ranklist_path):
                try:
                    with open(ranklist_path, 'r') as f_rank:
                        final_ranklist = json.load(f_rank)
                        if final_ranklist:
                            best_artifact_id = final_ranklist[0]
                            print(f"\nTop ranked artifact ID for '{ranked_loop.primary_logical_artifact_id}': {best_artifact_id}")
                        else:
                            print(f"Final ranklist for '{ranked_loop.primary_logical_artifact_id}' is empty.")
                except Exception as e:
                    print(f"Error reading or displaying final ranklist: {e}")
            else:
                print(f"Final ranklist file not found: {ranklist_path}")

        else:
            print(f"\nRanked agent loop for run '{final_state.get('run_id')}' ended with status: {final_state['status']}")
            if final_state.get("error"):
                print(f"Error: {final_state['error']}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a ranked agent loop with artifact management.")
    
    parser.add_argument("--query", type=str, required=True, help="The query or goal for the primary artifact")
    parser.add_argument("--agent-sequence", type=str, default="researcher,writer,editor", help="Comma-separated list of agent types")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations of the agent sequence")
    parser.add_argument("--max-steps-per-agent", type=str, default="3", help="Max steps per agent (integer or comma-separated list)")
    parser.add_argument("--model-id", type=str, default="gemini/gemini-2.0-flash", help="Model ID for main agents")
    parser.add_argument("--model-info-path", type=str, default="agents/utils/gemini/gem_llm_info.json", help="Path to model info JSON file")
    parser.add_argument("--use-custom-prompts", action="store_true", help="Use custom agent prompts")
    parser.add_argument("--enable-telemetry", action="store_true", help="Enable OpenTelemetry")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run ID to use or resume. If None, a new one is generated.")
    parser.add_argument("--load-run-state", action="store_true", help="If --run-id is provided, attempt to load its state and resume.")
    parser.add_argument("--ranking-llm-model-id", type=str, default="gemini/gemini-1.5-flash", help="Model ID for the ranking LLM judge")
    parser.add_argument("--max-ranklist-size", type=int, default=10, help="Max artifacts in a ranked list")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Ranking agent poll interval (seconds)")
    parser.add_argument("--primary-logical-artifact-id", type=str, default="final_report", help="Identifier for the main artifact being evolved")
    parser.add_argument("--run-data-base-dir", type=str, default="run_data", help="Base directory for run data")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Ensure .env is loaded if GEMINI_API_KEY is there
    from dotenv import load_dotenv
    load_dotenv()
    main() 