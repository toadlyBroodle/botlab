import threading
import time
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

# Avoid circular imports
if TYPE_CHECKING:
    from ..ranked_agent_loop import RankedAgentLoop

from smolagents import ToolCallingAgent
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from ..utils.agents.tools import apply_custom_agent_prompts, save_final_answer
from .tools import llm_judge

class RankingAgent:
    """A wrapper class for the ranking agent that evaluates and orders artifacts."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 3,
        model_id: str = "gemini/gemini-1.5-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3,
        max_ranklist_size: int = 10,
        poll_interval: float = 5.0,
        run_data_base_dir: str = "run_data",
        run_id: Optional[str] = None,
        logical_artifact_id: str = "final_report",
        parent_loop: Optional[Any] = None
    ):
        """Initialize the ranking agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            agent_description: Optional additional description to append to the base description
            system_prompt: Optional custom system prompt to use instead of the default
            max_steps: Maximum number of steps for the agent
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
            max_ranklist_size: Maximum number of artifact IDs to retain in each ranklist
            poll_interval: How often background thread checks for new metadata (seconds)
            run_data_base_dir: Base directory for storing run-specific data
            run_id: Specific ID for the run. If None, a new one is generated
            logical_artifact_id: The main conceptual artifact being evolved
        """
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
        
        # Store ranking configuration
        self.max_ranklist_size = max_ranklist_size
        self.poll_interval = poll_interval
        self.run_data_base_dir = run_data_base_dir
        self.logical_artifact_id = logical_artifact_id
        
        # Generate run_id if not provided
        if run_id is None:
            from datetime import datetime
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            print(f"Generated new run_id for RankingAgent: {self.run_id}")
        else:
            self.run_id = run_id
        
        # Set up paths
        self.run_dir = os.path.join(self.run_data_base_dir, self.run_id)
        self.metadata_path = os.path.join(self.run_dir, "metadata.jsonl")
        self.ranklist_path = os.path.join(self.run_dir, "ranking_state", f"{self.logical_artifact_id}.ranklist.json")
        self.ranking_state_dir = os.path.dirname(self.ranklist_path)
        
        # Ensure directories exist
        os.makedirs(self.ranking_state_dir, exist_ok=True)
            
        # Append additional description if provided
        base_description = 'A ranking agent that evaluates and orders artifacts based on quality and relevance to goals.'
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
        
        # Store parent loop reference for status updates
        self.parent_loop = parent_loop
        
        # Create the agent
        self.agent = ToolCallingAgent(
            tools=[],  # Tools would be defined here if needed
            model=self.model,
            name='ranking_agent',
            description=description,
            max_steps=max_steps,
        )

        # Default system prompt if none provided
        default_system_prompt = f"""You are a ranking agent responsible for evaluating and ordering artifacts based on their quality and relevance to specific goals.

Your primary tasks are:
1. Compare artifacts to determine which ones better accomplish the stated goal
2. Maintain a ranked list of artifacts based on these comparisons
3. Provide insights into why certain artifacts are ranked higher than others

When comparing artifacts, focus on:
- How well they address the stated goal
- Overall quality and coherence
- Accuracy and relevance of information
- Clarity and effectiveness of communication

Your logical artifact ID is: '{self.logical_artifact_id}'

Always be fair and consistent in your evaluations.
"""

        # Apply custom templates with the appropriate system prompt
        custom_prompt = system_prompt if system_prompt else default_system_prompt
        apply_custom_agent_prompts(self.agent, custom_prompt)
        
        # Initialize background thread components
        self.metadata_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self._last_processed_line = -1
        self.bg_thread = None
        
        print(f"RankingAgent for '{self.logical_artifact_id}' initialized (run_id: {self.run_id}).")
    
    def start_background_ranking(self):
        """Starts the background thread for continuous artifact monitoring and ranking."""
        if self.bg_thread is not None and self.bg_thread.is_alive():
            print("Background ranking thread is already running.")
            return
        
        # Initialize the ranklist file if it doesn't exist
        if not os.path.exists(self.ranklist_path):
            print(f"Initializing empty ranklist file for '{self.logical_artifact_id}'")
            self._write_ranklist([])
            
        self.shutdown_event.clear()  # Reset in case it was set previously
        self.bg_thread = threading.Thread(
            target=self._background_ranking_loop, 
            daemon=True
        )
        self.bg_thread.start()
        print(f"RankingAgent background thread for '{self.logical_artifact_id}' started.")
        
    def stop_background_ranking(self):
        """Stops the background thread for continuous artifact monitoring."""
        if self.bg_thread is None or not self.bg_thread.is_alive():
            print("No background ranking thread is running.")
            return
            
        print(f"Stopping RankingAgent background thread for '{self.logical_artifact_id}'...")
        self.shutdown_event.set()
        self.bg_thread.join(timeout=self.poll_interval * 2)
        if self.bg_thread.is_alive():
            print("Warning: RankingAgent background thread did not terminate gracefully.")
        else:
            print(f"RankingAgent background thread for '{self.logical_artifact_id}' stopped.")
    
    def rank_artifacts(self, prompt: str) -> str:
        """Manually triggers artifact evaluation and ranking.
        
        Args:
            prompt: The evaluation prompt containing goal and context
            
        Returns:
            A summary of the ranking results
        """
        # Time the execution
        start_time = time.time()
        
        # Parse the prompt to extract goal if possible
        goal = prompt
        if "goal:" in prompt.lower():
            for line in prompt.split("\n"):
                if line.lower().startswith("goal:"):
                    goal = line[5:].strip()
                    break
        
        # Get current ranking state
        current_ranklist = self._read_ranklist()
        
        # Build a response with the current ranking state
        response = f"Current Ranking for '{self.logical_artifact_id}':\n\n"
        
        if not current_ranklist:
            response += "No artifacts have been ranked yet.\n"
        else:
            # Load top 3 artifacts to provide details
            response += f"Top {min(3, len(current_ranklist))} artifacts:\n\n"
            for i, artifact_id in enumerate(current_ranklist[:3]):
                artifact_content = self._load_artifact_content(artifact_id)
                metadata = self._get_artifact_metadata(artifact_id)
                
                response += f"{i+1}. Artifact ID: {artifact_id}\n"
                if metadata:
                    response += f"   Created by: {metadata.get('agent_type', 'unknown')}\n"
                    response += f"   Iteration: {metadata.get('iteration', 'unknown')}\n"
                    response += f"   Timestamp: {metadata.get('timestamp', 'unknown')}\n"
                
                if artifact_content:
                    preview = artifact_content[:100].replace('\n', ' ')
                    response += f"   Preview: \"{preview}...\"\n\n"
                else:
                    response += f"   Content: [Could not load artifact content]\n\n"
            
            # Add total count
            response += f"Total artifacts ranked: {len(current_ranklist)}\n"
        
        # Manual comparison could be triggered here if specified in the prompt
        if "compare" in prompt.lower() and len(current_ranklist) >= 2:
            response += "\nComparing top two artifacts:\n"
            top_content = self._load_artifact_content(current_ranklist[0])
            second_content = self._load_artifact_content(current_ranklist[1])
            
            if top_content and second_content:
                winner, justification = llm_judge(top_content, second_content, goal, self.model)
                response += f"Result: {'First artifact' if winner == 'A' else 'Second artifact' if winner == 'B' else 'Equal quality'}\n"
                response += f"Justification: {justification}\n"
        
        # Calculate execution time
        execution_time = time.time() - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Save the result
        save_final_answer(
            agent=self.agent,
            result=response,
            query_or_prompt=prompt,
            agent_type="ranker"
        )
        
        return response
    
    def _background_ranking_loop(self):
        """The main loop of the ranking agent background thread."""
        print(f"RankingAgent background thread for '{self.logical_artifact_id}' started running.")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for new metadata entries
                new_artifacts = []
                current_line_num = -1
                
                if os.path.exists(self.metadata_path):
                    with self.metadata_lock:  # Lock for reading metadata
                        with open(self.metadata_path, 'r') as f:
                            for i, line in enumerate(f):
                                current_line_num = i
                                if i > self._last_processed_line:
                                    try:
                                        entry = json.loads(line.strip())
                                        if entry.get("logical_artifact_id") == self.logical_artifact_id:
                                            new_artifacts.append(entry)
                                            print(f"RankingAgent detected new artifact: {entry.get('artifact_id')} for {self.logical_artifact_id}")
                                    except json.JSONDecodeError:
                                        print(f"Warning: Skipping malformed line {i} in {self.metadata_path}")
                    
                    self._last_processed_line = current_line_num
                
                # Process new artifacts
                if new_artifacts:
                    current_list = self._read_ranklist()
                    needs_update = False
                    
                    for artifact_meta in new_artifacts:
                        print(f"RankingAgent processing: {artifact_meta.get('artifact_id')}")
                        modified, current_list = self._process_new_artifact(artifact_meta, current_list)
                        if modified:
                            needs_update = True
                    
                    if needs_update:
                        self._write_ranklist(current_list)
                        print(f"RankingAgent updated ranklist for {self.logical_artifact_id}: {current_list}")
                        
                        # Update parent loop state if available
                        if self.parent_loop:
                            self.parent_loop.update_ranking_status(new_ranklist=current_list)
            
            except Exception as e:
                print(f"Error in RankingAgent background loop for {self.logical_artifact_id}: {e}")
                # Avoid busy-waiting on persistent errors
                time.sleep(self.poll_interval * 2)
            
            # Wait before next check
            self.shutdown_event.wait(self.poll_interval)
        
        print(f"RankingAgent background thread for '{self.logical_artifact_id}' shutting down.")
    
    def _read_ranklist(self) -> List[str]:
        """Reads the current ranked list of artifact IDs."""
        if not os.path.exists(self.ranklist_path):
            return []
        
        try:
            with open(self.ranklist_path, 'r') as f:
                # Handle empty file case
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Ranklist file {self.ranklist_path} contains invalid JSON. Starting fresh.")
            return []
        except Exception as e:
            print(f"Error reading ranklist file {self.ranklist_path}: {e}")
            return []  # Return empty list on error
    
    def _write_ranklist(self, ranklist: List[str]):
        """Atomically writes the updated ranked list."""
        temp_path = self.ranklist_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(ranklist, f)
            os.replace(temp_path, self.ranklist_path)  # Atomic rename/replace
        except Exception as e:
            print(f"Error writing ranklist file {self.ranklist_path}: {e}")
            # Attempt to remove temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore errors during cleanup
    
    def _process_new_artifact(self, new_artifact_meta: Dict[str, Any], current_list: List[str]) -> Tuple[bool, List[str]]:
        """Compares and inserts a new artifact into the ranked list."""
        new_artifact_id = new_artifact_meta.get("artifact_id")
        if not new_artifact_id:
            print("Warning: New artifact metadata missing 'artifact_id'. Skipping.")
            return False, current_list
        
        if new_artifact_id in current_list:
            print(f"Artifact {new_artifact_id} already in ranklist. Skipping.")
            return False, current_list  # Already ranked
        
        print(f"Processing new artifact {new_artifact_id} for ranking...")
        new_artifact_content = self._load_artifact_content(new_artifact_id)
        if new_artifact_content is None:
            print(f"Warning: Could not load content for new artifact {new_artifact_id}. Skipping ranking.")
            return False, current_list
        
        # --- Adaptive Insertion Sort Logic ---
        
        # 1. Handle Empty List
        if not current_list:
            print(f"Ranklist empty. Adding {new_artifact_id} as the first item.")
            return True, [new_artifact_id]
        
        # 2. Compare vs. Top
        top_artifact_id = current_list[0]
        top_artifact_content = self._load_artifact_content(top_artifact_id)
        if top_artifact_content is not None:
            print(f"Comparing {new_artifact_id} vs. TOP {top_artifact_id}")
            # Goal description could be dynamic or passed in init
            goal = f"Improve the {self.logical_artifact_id}"
            winner, justification = llm_judge(new_artifact_content, top_artifact_content, goal, self.model)
            print(f"LLM Judge (vs Top): Winner={winner}, Justification: {justification}")
            
            # Record comparison result in parent loop state
            if self.parent_loop:
                comparison_record = {
                    "timestamp": time.time(),
                    "comparison_type": "new_vs_top",
                    "artifact_a": new_artifact_id,
                    "artifact_b": top_artifact_id,
                    "winner": winner,
                    "rationale": justification,
                    "goal": goal
                }
                self.parent_loop.update_ranking_status(new_comparison=comparison_record)
            
            if winner == 'A':  # New artifact is better than the current best
                print(f"{new_artifact_id} is better than current top {top_artifact_id}. Inserting at index 0.")
                current_list.insert(0, new_artifact_id)
                # Pruning applied after potential insertion
                if len(current_list) > self.max_ranklist_size:
                    current_list.pop()  # Remove the worst
                return True, current_list
            elif winner == 'Equal':
                # Insert after the top element if equal quality
                print(f"{new_artifact_id} is equal to current top {top_artifact_id}. Inserting at index 1.")
                current_list.insert(1, new_artifact_id)
                if len(current_list) > self.max_ranklist_size:
                    current_list.pop()
                return True, current_list
        else:
            print(f"Warning: Could not load content for top artifact {top_artifact_id}. Cannot compare.")
            # Decide on fallback behavior - maybe insert after top? Or append? Let's append for safety.
            current_list.append(new_artifact_id)
            if len(current_list) > self.max_ranklist_size:
                current_list.pop(0)  # Remove the presumed best if we can't compare
            return True, current_list
        
        # 3. Compare vs. Bottom (if not inserted at top) and list has more than 1 item
        if len(current_list) > 1:
            bottom_artifact_id = current_list[-1]
            bottom_artifact_content = self._load_artifact_content(bottom_artifact_id)
            if bottom_artifact_content is not None:
                print(f"Comparing {new_artifact_id} vs. BOTTOM {bottom_artifact_id}")
                goal = f"Improve the {self.logical_artifact_id}"
                winner, justification = llm_judge(new_artifact_content, bottom_artifact_content, goal, self.model)
                print(f"LLM Judge (vs Bottom): Winner={winner}, Justification: {justification}")
                
                # Record comparison result in parent loop state
                if self.parent_loop:
                    comparison_record = {
                        "timestamp": time.time(),
                        "comparison_type": "new_vs_bottom",
                        "artifact_a": new_artifact_id,
                        "artifact_b": bottom_artifact_id,
                        "winner": winner,
                        "rationale": justification,
                        "goal": goal
                    }
                    self.parent_loop.update_ranking_status(new_comparison=comparison_record)
                
                if winner == 'B':  # Bottom artifact is better than the new one
                    print(f"Bottom artifact {bottom_artifact_id} is better than {new_artifact_id}. Appending new artifact.")
                    current_list.append(new_artifact_id)
                    # Pruning check (although unlikely needed here if appended)
                    if len(current_list) > self.max_ranklist_size:
                        # This case shouldn't be hit often if max_ranklist_size > 1
                        current_list.pop(0)  # Remove best if appending exceeds limit
                    return True, current_list
            else:
                print(f"Warning: Could not load content for bottom artifact {bottom_artifact_id}. Cannot compare.")
                # Append for safety if comparison fails
                current_list.append(new_artifact_id)
                if len(current_list) > self.max_ranklist_size:
                    current_list.pop(0)
                return True, current_list
        
        # 4. Binary Search Insertion (if not inserted at top or bottom)
        # We know new artifact is worse than index 0 and potentially better than index -1
        low = 1  # Start searching from the second element
        high = len(current_list) - 1  # Up to the second to last element
        
        insertion_point = len(current_list)  # Default to append if search fails
        
        print(f"Starting binary search for {new_artifact_id} in indices [{low}, {high}]")
        
        while low <= high:
            mid = (low + high) // 2
            mid_artifact_id = current_list[mid]
            mid_artifact_content = self._load_artifact_content(mid_artifact_id)
            
            if mid_artifact_content is None:
                print(f"Warning: Could not load content for mid artifact {mid_artifact_id} during binary search. Aborting search for safety.")
                # Fallback: Append to avoid incorrect placement due to missing content
                insertion_point = len(current_list)
                break  # Exit the while loop
            
            print(f"Comparing {new_artifact_id} vs. index {mid} ({mid_artifact_id})")
            goal = f"Improve the {self.logical_artifact_id}"
            winner, justification = llm_judge(new_artifact_content, mid_artifact_content, goal, self.model)
            print(f"LLM Judge (Binary Search): Winner={winner}, Justification: {justification}")
            
            # Record comparison result in parent loop state
            if self.parent_loop:
                comparison_record = {
                    "timestamp": time.time(),
                    "comparison_type": "binary_search",
                    "artifact_a": new_artifact_id,
                    "artifact_b": mid_artifact_id,
                    "position": f"index_{mid}",
                    "winner": winner,
                    "rationale": justification,
                    "goal": goal
                }
                self.parent_loop.update_ranking_status(new_comparison=comparison_record)
            
            if winner == 'A' or winner == 'Equal':  # New is better or equal, search in the left half (lower indices)
                insertion_point = mid  # Potential insertion point found
                high = mid - 1
                print(f" -> New is better/equal. Search range now [{low}, {high}]")
            else:  # New is worse, search in the right half (higher indices)
                low = mid + 1
                print(f" -> New is worse. Search range now [{low}, {high}]")
        
        print(f"Binary search determined insertion point for {new_artifact_id}: {insertion_point}")
        current_list.insert(insertion_point, new_artifact_id)
        
        # 5. Pruning
        if len(current_list) > self.max_ranklist_size:
            print(f"Ranklist size {len(current_list)} exceeds max {self.max_ranklist_size}. Pruning worst.")
            current_list.pop()  # Remove the worst item (at the end after insertion)
        
        return True, current_list
    
    def _read_metadata(self) -> List[Dict[str, Any]]:
        """Reads all metadata entries from metadata.jsonl."""
        metadata = []
        if not os.path.exists(self.metadata_path):
            return metadata
        
        try:
            with self.metadata_lock:  # Acquire lock before reading
                with open(self.metadata_path, 'r') as f:
                    for line in f:
                        try:
                            metadata.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping malformed line in {self.metadata_path}")
        except Exception as e:
            print(f"Error reading metadata file {self.metadata_path}: {e}")
        
        return metadata
    
    def _get_artifact_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Gets metadata for a specific artifact ID."""
        metadata_entries = self._read_metadata()
        for entry in metadata_entries:
            if entry.get("artifact_id") == artifact_id:
                return entry
        return None
    
    def _load_artifact_content(self, artifact_id: str) -> Optional[str]:
        """Loads artifact content given its ID by looking up path in metadata."""
        metadata_entries = self._read_metadata()
        for entry in metadata_entries:
            if entry.get("artifact_id") == artifact_id:
                relative_path = entry.get("relative_path")
                if relative_path:
                    full_path = os.path.join(self.run_dir, relative_path)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    except FileNotFoundError:
                        print(f"Error: Artifact file not found at {full_path} for ID {artifact_id}")
                        return None
                    except Exception as e:
                        print(f"Error reading artifact file {full_path}: {e}")
                        return None
        
        print(f"Warning: Artifact ID {artifact_id} not found in metadata.")
        return None
    
    def compare_files(self, file_path_a: str, file_path_b: str, goal: Optional[str] = None) -> Tuple[str, str]:
        """Compares two files and returns the winner and rationale.
        
        Args:
            file_path_a: Path to the first file
            file_path_b: Path to the second file
            goal: Optional goal for the comparison. If None, uses the default goal.
            
        Returns:
            Tuple of (winner, rationale) where winner is 'A', 'B', or 'Equal'
        """
        print(f"Comparing files:\nA: {file_path_a}\nB: {file_path_b}")
        
        # Read the file contents
        try:
            with open(file_path_a, 'r', encoding='utf-8') as f:
                content_a = f.read()
        except Exception as e:
            error_msg = f"Error reading file A ({file_path_a}): {e}"
            print(error_msg)
            return "Error", error_msg
            
        try:
            with open(file_path_b, 'r', encoding='utf-8') as f:
                content_b = f.read()
        except Exception as e:
            error_msg = f"Error reading file B ({file_path_b}): {e}"
            print(error_msg)
            return "Error", error_msg
        
        # Use default goal if none provided
        if goal is None:
            goal = f"Improve the {self.logical_artifact_id}"
        
        # Use the existing llm_judge function to compare
        start_time = time.time()
        winner, rationale = llm_judge(content_a, content_b, goal, self.model)
        end_time = time.time()
        
        print(f"Comparison completed in {end_time - start_time:.2f}s")
        print(f"Winner: {'File A' if winner == 'A' else 'File B' if winner == 'B' else 'Equal'}")
        print(f"Rationale: {rationale[:200]}...")  # Print truncated rationale
        
        # Record comparison in parent loop state if available
        if self.parent_loop:
            comparison_record = {
                "timestamp": time.time(),
                "comparison_type": "direct_file_comparison",
                "file_a": file_path_a,
                "file_b": file_path_b,
                "winner": winner,
                "rationale": rationale,
                "goal": goal
            }
            self.parent_loop.update_ranking_status(new_comparison=comparison_record)
        
        return winner, rationale
    
    def __del__(self):
        """Clean up resources when the agent is garbage collected."""
        self.stop_background_ranking() 

def main():
    """Command-line interface for comparing two files directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two files and rank them.')
    parser.add_argument('file_a', type=str, help='Path to the first file')
    parser.add_argument('file_b', type=str, help='Path to the second file')
    parser.add_argument('--goal', type=str, default=None, help='Goal for comparison')
    parser.add_argument('--model', type=str, default="gemini/gemini-1.5-flash", help='Model ID to use')
    parser.add_argument('--model-info-path', type=str, default="agents/utils/gemini/gem_llm_info.json", help='Path to model info JSON')
    parser.add_argument('--output', type=str, default=None, help='Optional file to write full output to')
    
    args = parser.parse_args()
    
    # Create a standalone ranking agent
    from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
    
    model = RateLimitedLiteLLMModel(
        model_id=args.model,
        model_info_path=args.model_info_path,
        base_wait_time=2.0,
        max_retries=3
    )
    
    ranking_agent = RankingAgent(model=model)
    
    # Compare the files
    winner, rationale = ranking_agent.compare_files(args.file_a, args.file_b, args.goal)
    
    # Print the result
    print("\n" + "="*80)
    print(f"WINNER: {'File A' if winner == 'A' else 'File B' if winner == 'B' else 'Equal'}")
    print("="*80)
    print("RATIONALE:")
    print(rationale)
    print("="*80)
    
    # Write to output file if specified
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"File A: {args.file_a}\n")
                f.write(f"File B: {args.file_b}\n")
                f.write(f"Goal: {args.goal or 'Default'}\n")
                f.write(f"Winner: {'File A' if winner == 'A' else 'File B' if winner == 'B' else 'Equal'}\n\n")
                f.write("Rationale:\n")
                f.write(rationale)
            print(f"Full results written to {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main() 