# Agent Systems

This directory contains agent systems built with smolagents:

- **researcher**: A web search and information gathering agent
- **writer-critic**: A creative writing system with feedback and iteration
- **editor**: A content editing system with fact checking capabilities
- **manager**: A flexible coordination agent that can manage multiple specialized agents

## 📦 Installation

```bash
# From the root directory
./setup_env.sh
source .venv/bin/activate
cd agents/
echo "GEMINI_API_KEY=<your-api-key>" >> .env
```

## 🏃‍♀️ Running Agents

Run the agents from the activated virtual environment:

```bash
# Make sure the virtual environment is activated
source .venv/bin/activate

# Run from project root directory
python -m agents.researcher.example --query "What are the latest advancements in quantum computing?"

# Or run from agents directory
cd agents
python -m researcher.example --query "What are the latest advancements in quantum computing?"
```

## 📚 Use as a Submodule

This repository is designed to be usable as a submodule in other projects:

```bash
# In your project
git submodule add https://github.com/yourusername/botlab.git
cd botlab
./setup_env.sh
```

Then import agents with absolute imports:
```python
from botlab.agents import AgentLoop
from botlab.agents.researcher.agents import ResearcherAgent
```

## 🤖 Available Agents

### 🔍 Researcher Agent

The researcher agent can search the web, extract information from websites, and compile research reports.

```bash
# Run the researcher agent
python -m agents.researcher.example --query "What are the latest advancements in quantum computing?"
```

Key features:
- Web search via DuckDuckGo
- Web page content extraction
- arXiv academic paper search
- PDF to markdown conversion for research papers

### ✍️ Writer-Critic Agent

The writer-critic system consists of two agents working together: a writer that drafts content and a critic that provides feedback.

```bash
# Run the writer-critic system
python -m agents.writer_critic.main --prompt "Write a short story about a robot who discovers emotions."
```

Key features:
- Creative content generation
- Iterative refinement through feedback
- Customizable writing styles and critique approaches

### 📝 Editor Agent

The editor agent improves content quality while ensuring factual accuracy through fact checking.

```bash
# Run the editor agent
python -m agents.editor.main --content "Content to edit and fact check"
```

Key features:
- Content editing and proofreading
- Fact checking with source verification
- Clarity and style improvements
- Maintains original content style and intent

### 💼 Manager Agent

The manager agent coordinates multiple specialized agents to solve complex tasks.

```bash
# Run the manager with researcher, writer, and editor agents
python -m agents.manager.main --managed-agents "researcher,writer,editor" --query "Write an article about recent advances in quantum computing."
```

Key features:
- Orchestrates multiple specialized agents
- Delegates subtasks to appropriate agents
- Combines results into a cohesive output

## 🛠️ Advanced Configuration

### Agent Configuration

You can customize agent behavior using JSON configuration files or command-line flags:

```bash
# Using a config file
python -m agents.manager.main --managed-agents researcher,writer,editor --config-file agent_configs.json

# Using the simplified custom prompts flag
python -m agents.manager.main --managed-agents researcher,writer,editor --use-custom-prompts --query "Write an article about recent advances in quantum computing."
```

The `--use-custom-prompts` flag enables a set of predefined custom descriptions and prompts for all agents, which are defined in the example.py file. This provides a simpler alternative to specifying each property individually.

Example configuration format for config files:
```json
{
  "researcher_description": "Expert researcher with focus on scientific papers",
  "researcher_prompt": "You are a meticulous researcher who prioritizes academic sources",
  "writer_description": "Creative writer with journalistic style",
  "writer_prompt": "Write engaging content with a focus on clarity and accuracy",
  "critic_description": "Detail-oriented editor with high standards",
  "critic_prompt": "Evaluate writing for clarity, accuracy, and engagement",
  "editor_description": "Skilled editor with focus on accuracy and clarity",
  "editor_prompt": "Edit content to ensure factual accuracy while maintaining style",
  "fact_checker_description": "Thorough fact checker with attention to detail",
  "fact_checker_prompt": "Verify claims against reliable sources with precision"
}
```

### Customization Best Practices

When customizing agent behavior:

1. **Preserve main.py files**: The main.py files should generally be left as is, as they provide the core functionality and interface for each agent system.

2. **Create custom example files**: Instead of modifying the original example.py files, create new versions (e.g., my_custom_example.py) that import from main.py and implement your specific customizations.

3. **Use configuration options**: Leverage the existing configuration options (like --use-custom-prompts or --config-file) rather than modifying the core code.

4. **Share custom configurations**: Custom agent descriptions and prompts should be defined in your custom example files, following the pattern in example.py.

This approach ensures that the core functionality remains stable while allowing for flexible customization.

### Model Configuration

You can specify which LLM to use and configure rate limiting:

```bash
# Using a specific model
python -m agents.researcher.main --model-id "gemini/gemini-2.0-pro" --query "What is quantum computing?"
```

## Start phoenix telemetry server
To monitor the agents telemetry, you can start the telemetry server with:

```bash
python -m phoenix.server.main serve
```

To view the telemetry data, open the phoenix UI at [http://localhost:6006](http://localhost:6006).

Also make sure to add `--telemetry` to cli agent calls to enable their telemetry.

## Running the Examples

### Option 1: Using the run_examples.py script

From the `agents` directory:

```bash
# Run the researcher example
python run_examples.py researcher "your query"

# Run the basic manager example (with researcher agent)
python run_examples.py manager "your query"

# Run the advanced manager example (with custom agent configuration)
python run_examples.py manager-advanced "your query"

# Run the manager with custom agent selection
python run_examples.py manager-custom "your query" --agents researcher writer editor

# Run the writer-critic example
python run_examples.py writer

# Run the editor example
python run_examples.py editor "content to edit and fact check"

# Run all examples
python run_examples.py all

# Enable telemetry for any example
python run_examples.py researcher "your query" --telemetry
```

### Option 2: Running individual examples as modules

From the `agents` directory:

```bash
# Run the researcher example
python -m researcher.example "your query"

# Run the writer-critic example
python -m writer_critic.example

# Run the editor example
python -m editor.example "content to edit and fact check"

# Run the manager example with default settings
python -m manager.example --query "your query" --managed-agents researcher,writer,editor

# Run the manager with custom prompts
python -m manager.example --query "your query" --managed-agents researcher,writer,editor --use-custom-prompts
```

## Project Structure

The agents project follows this structure:

```
agents/
    |__ researcher/         # Researcher agent project
    |__ utils/              # Shared utilities
    |   |__ gemini/         # Google Gemini LLM API utilities
    |__ writer_critic/      # Writer-Critic agent project
    |__ editor/             # Editor agent project
    |__ manager/            # Manager agent project
```

Each agent project (researcher, writer_critic, manager) contains these common files:
- `__init__.py`
- `agents.py`: Project agents
- `example.py`: Example usages
- `main.py`: Main setup and entry point
- `tools.py`: Tools used by the agents

When creating new agents, it's probably best to reuse this structure, and modify the `main.py`, `agents.py`, and `tools.py` files. `example.py` serves as a good intro example for how to use the agent.
