# 🤖🧪 botlab: a collection of AI bots, agents, and teams

A laboratory for experimenting with AI agents and automation tools. Built with Python and modern AI frameworks.

## ⭐ Features

- 🤖 **AI Agents**: Modern LLM-powered agents to automate nearly any job
- 🤝 **Agent Teams**: Teams of agents that collaborate to solve complex problems
- 🧩 **Modular Architecture**: Plug-and-play components for custom AI solutions
- ✨ **Simplicity**: Minimal dependencies and easy to understand code
- 📊 **Extensive Logging**: Built-in monitoring and debugging tools
- 📁 **Simple File Management**: Agent outputs stored in *data/* directories

## 📚 Projects

This repository contains two main projects:

### 1. 🧠 Agents (smolagent framework)

The `agents/` directory contains a collection of AI agents built with the smolagent framework. These agents can perform tasks like research, writing, and coordination.

```bash
# Setup the environment first
./setup_env.sh
source .venv/bin/activate

# Example: Run the researcher agent
python -m agents.researcher.example --query "What are the latest advancements in AI?"
```

**[View the full Agents documentation →](agents/README.md)**

Key components:
- **ResearcherAgent**: Web search and information gathering
- **WriterAgent & CriticAgent**: Creative writing with feedback
- **EditorAgent & FactCheckerAgent**: Content editing with fact checking
- **ManagerAgent**: Coordinates multiple specialized agents
- **QAQCAgent**: Compares outputs and selects the best one
- **AgentLoop**: Orchestrates iterative workflows between multiple agents

### 2. 🐝 Swarms (OpenAI's swarm framework)

The `swarms/` directory contains implementations based on OpenAI's swarm framework, allowing for the creation of collaborative agent systems.

```bash
# Setup the environment first (if not already done)
./setup_env.sh
source .venv/bin/activate

# Example: Run the writer-critic swarm
python -m swarms.writer-critic.writer-critic
```

**[View the full Swarms documentation →](swarms/README.md)**

Key features:
- **Writer-Critic System**: Collaborative writing with feedback loops
- **Multi-agent collaboration**: Agents working together on complex tasks
- **Emergent behavior**: Solutions that arise from agent interactions
- **Scalable architecture**: Add more agents to tackle larger problems

## 🛠️ Setup

The project uses a single virtual environment at the root directory for all components.

```bash
# Setup the environment
./setup_env.sh
source .venv/bin/activate

# Set API keys in the appropriate .env files
# For agents: add GEMINI_API_KEY to agents/.env
# For swarms: add OPENAI_API_KEY to swarms/.env
```

For detailed instructions, please refer to the [INSTALL.md](INSTALL.md) file.

## 📦 Using Botlab in Other Projects

You can easily use botlab agents in your own projects by adding it as a Git submodule:

```bash
# Add botlab as a submodule to your project
cd your-project
git submodule add https://github.com/yourusername/botlab.git
git commit -m "Add botlab as submodule"

# Setup the environment
cd botlab
./setup_env.sh
```

Then in your Python code:

```python
import os
import sys
from dotenv import load_dotenv

# Add botlab to Python path
sys.path.append("./botlab")

# Import the agent you need
from botlab.agents.researcher.agents import ResearcherAgent

# Setup and use the agent
load_dotenv()
researcher = ResearcherAgent()
result = researcher.run_query("Your query here")
```

To update the submodule when botlab changes:

```bash
git submodule update --remote botlab
git commit -m "Update botlab submodule"
```

## 🤝 Contributing

1. Fork and clone the repository
    - `git clone https://github.com/yourusername/botlab.git`
2. Create a new branch
3. Submit a pull request

## 📜 License

GNU General Public License v3.0 - See [LICENSE](LICENSE)

---

❤️ Thank you for using botlab! We hope this project helps you harness the awesome power of AI.
