# 🤖🧪 botlab: a collection of AI bots, agents, and teams

A laboratory for experimenting with AI agents and automation tools. Built with Python and modern AI frameworks.

## ⭐ Features

- 🤖 **AI Agents**: Modern LLM-powered agents to automate nearly any job
- 🤝 **Agent Teams**: Teams of agents that collaborate to solve complex problems
- 🧩 **Modular Architecture**: Plug-and-play components for custom AI solutions
- ✨ **Simplicity**: Minimal dependencies and easy to understand code
- 📊 **Extensive Logging**: Built-in monitoring and debugging tools

## 📚 Projects

This repository contains two main projects:

### 1. 🧠 Agents (smolagent framework)

The `agents/` directory contains a collection of AI agents built with the smolagent framework. These agents can perform tasks like research, writing, and coordination.

```bash
# Example: Run the researcher agent
cd agents
poetry run python -m researcher.example --query "What are the latest advancements in AI?"
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
# Example: Run the writer-critic swarm
cd swarms
python -m writer-critic.writer-critic
```

**[View the full Swarms documentation →](swarms/README.md)**

Key features:
- **Writer-Critic System**: Collaborative writing with feedback loops
- **Multi-agent collaboration**: Agents working together on complex tasks
- **Emergent behavior**: Solutions that arise from agent interactions
- **Scalable architecture**: Add more agents to tackle larger problems

## 🛠️ Setup

Each project maintains it's own virtual environment and dependencies.

### Agents Project Setup

```bash
cd agents
poetry install
# Set GEMINI_API_KEY in .env
```

### Swarms Project Setup

```bash
cd swarms
poetry install
# Set OPENAI_API_KEY in .env
```

For detailed instructions, please refer to the respective README files in each directory.

## 🤝 Contributing

1. Fork and clone the repository
    - `git clone https://github.com/yourusername/botlab.git`
2. Create a new branch
3. Submit a pull request

## 📜 License

GNU General Public License v3.0 - See [LICENSE](LICENSE)

---

❤️ Thank you for using botlab! We hope this project helps you harness the awesome power of AI.
