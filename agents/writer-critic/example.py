import time
import os
from agents.writer_critic.main import initialize

def main():
    # Initialize writer-critic system with 5 steps
    print("Initializing writer-critic system...")
    run_writing_task = initialize(max_steps=5)
    
    # Example writing prompt
    prompt = """Write a book outline for a sci-fi story about humans using AGI-powered robots to colonize the Moon. 
    Include at least 5 main plot points, 3-5 key characters with brief descriptions, 3 major themes, 
    1-2 premise-shaking twists, a few subplots, a few minor characters, and chapters with detailed 
    summaries of the flow of the story. Assume that AGI has helped humans develop compact fusion generators, 
    supplying intelligent life with unfathomable amounts of energy for terraforming."""
    
    # Time the execution
    print("Starting the writing task. This may take a few minutes...")
    start_time = time.time()
    
    # Run the writing task
    final_draft = run_writing_task(prompt)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # Save final draft to file in the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "final_draft.md")
    with open(output_path, "w") as f:
        f.write(final_draft)
    
    print(f"\nFinal draft saved to {output_path}")
    print("\nDraft versions were saved in the drafts directory during the process.")

if __name__ == "__main__":
    main() 