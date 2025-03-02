# Swarms

This directory contains implementations based on OpenAI's swarm framework, allowing for the creation of collaborative agent systems.

## Install dependencies

```bash
cd botlab

poetry install

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=<your-api-key>" >> .env
```

## Available Swarm Systems

### Writer-Critic System

The Writer-Critic system is a collaborative AI writing system that uses two specialized agents working together to create and refine creative content:

- **Writer Agent**: Creates and revises drafts based on feedback
- **Critic Agent**: Provides detailed feedback on drafts to improve quality

#### Features

- **Iterative Refinement**: Multiple rounds of writing and critique to improve content quality
- **Automatic Draft Saving**: Saves each iteration of the draft for review
- **Detailed Logging**: Tracks the conversation and progress between agents
- **Configurable Iteration Limit**: Set maximum number of revision cycles
- **Word Count Tracking**: Monitors the length of each draft

#### Usage

```bash
# Run the writer-critic system to create a book outline
cd swarms
python -m writer-critic.writer-critic
```

#### Configuration

The Writer-Critic system can be customized by modifying the following parameters in the script:

- **Models**: Change the LLM models used by each agent (currently using gpt-4o-mini)
- **Iteration Limit**: Adjust the maximum number of revision cycles (default: 5)
- **Agent Instructions**: Customize the prompts for the writer and critic agents
- **Output Directory**: Change where drafts and logs are saved

#### Example Agent Instructions

**Writer Agent**:
```
You are a uniquely talented, often eccentric, esoteric, science fiction writer tasked with creating a riveting story about humans and AGI-powered robots to colonize the Moon. Your writing is vivid, engaging, and scientifically plausible...
```

**Critic Agent**:
```
You are an insightful, brutally honest literary critic with expertise in science fiction. Your role is to analyze the story's structure, themes, character arcs, and scientific elements...
```

#### Output Files

The system generates several output files:

- **Draft Files**: Saved in the `drafts/` directory with iteration numbers
- **Conversation Logs**: Detailed logs of the agent interactions in `logs/`
- **Message Logs**: Complete message history in `logs/`

## Other Swarm Examples

I will probably add more examples in the future, however to be honest I've found the smolagent framework to be more powerful and flexible for building complex agent systems, so my focus remains on that project for now.
