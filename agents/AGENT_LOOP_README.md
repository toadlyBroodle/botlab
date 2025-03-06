# Agent Loop

A flexible script for looping calls to multiple agents in sequence, with state management and result passing between agents.

## Overview

The Agent Loop script provides a framework for:

1. Running a sequence of agents in order (e.g., researcher → writer → editor → qaqc)
2. Passing results between agents
3. Running multiple iterations of the sequence
4. Persisting state between runs
5. Customizing agent prompts and behavior
6. Quality control through QAQC agent comparison of outputs

This allows for complex workflows where agents build on each other's work and iteratively improve results.

## Usage

### Basic Usage

Run the agent loop with default settings:

```bash
cd agents/
poetry run python agent_loop.py --query "Explain the impact of artificial intelligence on modern healthcare"
```

This will:
1. Initialize a researcher, writer, and editor agent
2. Run them in sequence for 3 iterations
3. Print the final result from the editor agent

### Example Script with QAQC

For a simpler example with QAQC agent included in the sequence:

```bash
cd agents/
poetry run python agent_loop_example.py --query "Explain the impact of artificial intelligence on modern healthcare"
```

### Command Line Options

The agent loop script supports the following command line options:

```
--query TEXT                  The query to process [required]
--agent-sequence TEXT         Comma-separated list of agent types to call in sequence
                              [default: researcher,writer,editor]
--max-iterations INTEGER      Maximum number of iterations through the entire sequence
                              [default: 3]
--max-steps-per-agent INTEGER Maximum steps for each agent [default: 15]
--max-retries INTEGER         Maximum retries for rate limiting [default: 3]
--model-id TEXT               The model ID to use [default: gemini/gemini-2.0-flash]
--model-info-path TEXT        Path to model info JSON file
                              [default: utils/gemini/gem_llm_info.json]
--use-custom-prompts          Whether to use custom agent descriptions and prompts
--enable-telemetry            Whether to enable OpenTelemetry tracing
--state-file TEXT             Path to a file for persisting state between runs
```

### Available Agent Types

The following agent types are available:

- `researcher`: Searches the web and gathers information
- `writer`: Creates content based on research
- `editor`: Edits and fact-checks content
- `qaqc`: Compares outputs and selects the best one

## Examples

### Custom Agent Sequence with QAQC

Run a sequence with researcher, writer, and QAQC:

```bash
poetry run python agent_loop.py --query "Explain quantum computing" --agent-sequence "researcher,writer,qaqc"
```

### Multiple Iterations with QAQC

Run 5 iterations of the full sequence with QAQC:

```bash
poetry run python agent_loop.py --query "Write a short story about AI" --max-iterations 5 --agent-sequence "researcher,writer,editor,qaqc"
```

### State Persistence

Run with state persistence to allow resuming from where you left off:

```bash
poetry run python agent_loop.py --query "Explain climate change" --state-file "climate_change_state.json"
```

If the script is interrupted, you can resume by running the same command again.

### Custom Prompts

Use custom agent descriptions and prompts:

```bash
poetry run python agent_loop.py --query "Explain blockchain technology" --use-custom-prompts
```

### QAQC at Different Positions

You can place the QAQC agent at different positions in the sequence:

```bash
# QAQC after writer but before editor
poetry run python agent_loop.py --query "Explain machine learning" --agent-sequence "researcher,writer,qaqc,editor"

# Multiple QAQC agents at different stages
poetry run python agent_loop.py --query "Explain machine learning" --agent-sequence "researcher,qaqc,writer,qaqc,editor"
```

## How It Works

The Agent Loop script:

1. Initializes each agent in the sequence
2. For each iteration:
   - Runs each agent in order
   - Passes results from previous agents to the next agent
   - Stores results in the state
   - If a QAQC agent is in the sequence, it compares outputs from the previous iteration with the current one
   - The QAQC agent can select the better output and update the results accordingly
3. After completing all iterations, returns the final result

The state is persisted to a file if specified, allowing for resuming interrupted runs.

## QAQC Agent

The QAQC (Quality Assurance/Quality Control) agent is a special agent that:

1. Compares outputs from different iterations
2. Analyzes them based on quality, accuracy, completeness, and relevance
3. Selects the best output to continue with
4. Provides detailed feedback on why one output is better than the other
5. Includes a structured decision format for automated processing: `DECISION: SELECT OUTPUT X`
6. Uses the `select_best_output` tool to programmatically select the best output

When placed in the agent sequence, the QAQC agent will automatically compare the output from the agent immediately before it across iterations. If it determines that the previous iteration's output was better, it will update the current iteration's result to use that better output.

This helps ensure that each iteration builds on the best previous work, rather than just the most recent output.

### QAQC Decision Format

The QAQC agent uses a structured format to indicate its decision:

```
DECISION: SELECT OUTPUT 1
```

or

```
DECISION: SELECT OUTPUT 2
```

This format allows the agent loop to programmatically extract the decision and take appropriate action. The agent loop first looks for this structured format, and if not found, falls back to a heuristic approach based on the content of the QAQC agent's output.

### QAQC Implementation Details

The QAQC agent is implemented as a standard `CodeAgent` with the following key components:

1. **Tool Usage**: It uses the `select_best_output` tool to programmatically select the best output, which returns a structured result containing the selected output number and text.

2. **Memory Access**: The agent's tool calls are accessed through the `memory.steps` attribute, with robust checks to ensure compatibility with different agent implementations:
   ```python
   if hasattr(qaqc_agent, 'memory') and hasattr(qaqc_agent.memory, 'steps'):
       for step in qaqc_agent.memory.steps:
           if hasattr(step, 'tool_calls') and step.tool_calls:
               # Process tool calls
   ```

3. **Fallback Mechanism**: If no tool call is found or if the tool call doesn't contain the expected information, the agent falls back to a default selection to ensure robustness.

This implementation follows the DRY (Don't Repeat Yourself) principle by leveraging the existing agent framework and adding only the specific functionality needed for QAQC operations.

## Extending

To extend the Agent Loop script:

1. Add new agent types by implementing them in the standard agent structure
2. Modify the `_format_prompt_for_agent` method to handle new agent types
3. Add custom logic for determining when to stop iterations
4. Enhance the QAQC agent's selection logic for specific use cases
5. To extend the QAQC agent:
   - Add new tools to `qaqc/tools.py` for additional functionality
   - Modify the `create_qaqc_agent` function in `qaqc/agents.py` to include new tools
   - Update the `run_qaqc_comparison` function to handle different output formats
   - Customize the system prompt to include instructions for new tools or decision formats

## Troubleshooting

If you encounter issues:

1. Check that your GEMINI_API_KEY is set correctly
2. Ensure you're running from the `agents/` directory
3. Check the state file for errors if using state persistence
4. Try running with a single agent to isolate issues
5. If using QAQC, make sure it has at least one agent before it in the sequence
6. If the QAQC agent is not selecting outputs correctly:
   - Verify that the `select_best_output` tool is properly imported in `qaqc/agents.py`
   - Check that the agent's prompt includes instructions to use the tool
   - Ensure the agent has access to the memory attribute by adding appropriate checks
   - Review the comparison request format to make sure it clearly instructs the agent to make a selection
