# Agent Systems

This directory contains agent systems built with smolagents:

- **researcher**: A web search and information gathering agent
- **writer-critic**: A creative writing system with feedback and iteration
- **editor**: A content editing system with fact checking capabilities
- **manager**: A flexible coordination agent that can manage multiple specialized agents

## 📦 Installation

```bash
cd agents/
poetry install
echo "GEMINI_API_KEY=<your-api-key>" >> .env
```

## 🤖 Available Agents

### 🔍 Researcher Agent

The researcher agent can search the web, extract information from websites, and compile research reports.

```bash
# Run the researcher agent
poetry run python -m researcher.main --query "What are the latest advancements in quantum computing?"
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
poetry run python -m writer_critic.main --prompt "Write a short story about a robot who discovers emotions."
```

Key features:
- Creative content generation
- Iterative refinement through feedback
- Customizable writing styles and critique approaches

### 📝 Editor Agent

The editor agent improves content quality while ensuring factual accuracy through fact checking.

```bash
# Run the editor agent
poetry run python -m editor.main --content "Content to edit and fact check"
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
poetry run python -m manager.main --managed-agents "researcher,writer,editor" --query "Write an article about recent advances in quantum computing."
```

Key features:
- Orchestrates multiple specialized agents
- Delegates subtasks to appropriate agents
- Combines results into a cohesive output

## 🛠️ Advanced Configuration

### Agent Configuration

You can customize agent behavior using JSON configuration files or command-line arguments:

```bash
# Using a config file
poetry run python -m manager.main --managed-agents researcher,writer,editor --config-file agent_configs.json

# Using a JSON string
poetry run python -m manager.main --managed-agents researcher,writer,editor --agent-configs '{"researcher_description": "Expert researcher", "writer_prompt": "Write in a journalistic style", "editor_description": "Skilled editor with focus on accuracy and clarity", "editor_prompt": "Edit content to ensure factual accuracy while maintaining style"}'
```

Example configuration format:
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

### Model Configuration

You can specify which LLM to use and configure rate limiting:

```bash
# Using a specific model
poetry run python -m researcher.main --model-id "gemini/gemini-2.0-pro" --query "What is quantum computing?"
```

## Start phoenix telemetry server
To monitor the agents telemetry, you can start the telemetry server with:

```bash
poetry run python -m phoenix.server.main serve
```

To view the telemetry data, open the phoenix UI at [http://localhost:6006](http://localhost:6006).

Also make sure to add `--telemetry` to cli agent calls to enable their telemetry.

## Running the Examples

### Option 1: Using the run_examples.py script

From the `agents` directory:

```bash
# Run the researcher example
poetry run python run_examples.py researcher "your query"

# Run the basic manager example (with researcher agent)
poetry run python run_examples.py manager "your query"

# Run the advanced manager example (with custom agent configuration)
poetry run python run_examples.py manager-advanced "your query"

# Run the manager with custom agent selection
poetry run python run_examples.py manager-custom "your query" --agents researcher writer

# Run the writer-critic example
poetry run python run_examples.py writer

# Run all examples
poetry run python run_examples.py all

# Enable telemetry for any example
poetry run python run_examples.py researcher "your query" --telemetry
```

### Option 2: Running individual examples as modules

From the `agents` directory:

```bash
# Run the researcher example
poetry run python -m researcher.example "your query"

# Run the writer-critic example
poetry run python -m writer_critic.example

# Run the manager example
poetry run python -m manager.example "your query"


# Run the manager with specific agent types
poetry run python -m manager.example "your query" --managed-agents researcher,writer
```

## Project Structure

The agents project follows this structure:

```
agents/
    |__ .venv/              # Poetry managed virtual environment
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
