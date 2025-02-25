# Writer-Critic Agent System

An AI creative writing system using smolagents to implement a writer-critic workflow for iterative content creation. This system uses a manager-worker pattern where a writer agent creates content and then leverages a critic agent to receive feedback and improve the drafts.

## Key Components

- **Writer Agent**: Creates drafts and manages the improvement process
- **Critic Agent**: Provides detailed feedback on drafts
- **Draft Saving Tool**: Saves each version of the draft for tracking progress

## How It Works

1. The writer agent creates an initial draft based on a prompt
2. The draft is saved using the save_draft tool
3. The writer agent calls the critic agent directly to get feedback
4. The critic agent provides detailed analysis and suggestions
5. The writer creates a revised draft incorporating the feedback
6. This process repeats until the content is satisfactory

## Usage

```python
from writer_critic.main import initialize

# Initialize with 5 steps (iterations)
run_writing_task = initialize(max_steps=5)

# Run with a writing prompt
result = run_writing_task("Write a story about...")

# Use the final draft
print(result)
```

## Example

See `example.py` for a complete usage example. 