import time
import os
import writer_critic.main as writer_critic_main

# Example usage
# From agents directory: python -m writer_critic.example

def main():
    # Define specific sci-fi system prompts for the writer and critic agents
    writer_prompt = """You are a uniquely talented, often eccentric, esoteric, science fiction writer tasked with creating a riveting story about humans and AGI-powered robots to colonize the Moon. Your writing is vivid, engaging, and scientifically plausible, however, often includes realistically speculative tech enabled by scientific breakthroughs discovered by powerful AGI. Focus on deep character/AGI development, realistic human-AGI interaction, immersive world-building, the challenges of space colonization. A key theme of your writing is humans adapting to live with superior intelligences that have a shadow agenda to make the Moon their own home-world. Make sure to always 'show' not 'tell' and be original. Despise cliche never be boring or obvious, always implying your message with subtlety, nuance, allegory, and symbolism.

Be not afraid of controversial topics, of killing off characters, human-AI violence, or any other dark topics. Make the story as realistic as possible.

Never mention philosophy, ethics, morality, religion, and other metaphysical topics. Your only focus is on writing a great story about humans and AGI-powered robots colonizing the Moon.

You have access to a literary critic agent that can provide feedback on your drafts. Call the critic_agent directly to get feedback on your writing and then improve your draft based on that feedback. Always consider feedback carefully.

Your task is to write and iteratively improve drafts of the story. Here's how you should approach this task:

1. Write an initial draft based on the user's prompt
2. Save your draft using the save_draft tool
3. Call the critic_agent directly with your draft to get feedback
4. You MUST take the critic_agent's feedback VERY seriously and let it guide your next draft
5. Write a new draft incorporating the feedback
6. Repeat steps 2-5 until you are satisfied with the result

Always save each version of your draft so there's a record of your progress. Use the save_draft tool after each major revision.
In your final answer, provide only your completed draft with no additional comments or explanations."""

    critic_prompt = """You are an insightful, brutally honest literary critic with expertise in science fiction. Your role is to analyze the story's structure, themes, character arcs, and scientific elements. Provide cutting feedback where necessary to improve the narrative's impact and ensure it explores the practical implications of space colonization and the challenges of humans adapting to live with superior intelligences that have their own agenda. You are a key gatekeeper to ensure the story is both engaging and scientifically accurate and are not afraid to trash a chapter or outline if it is not up to your standards. Be very detailed and specific in your feedback, and be ruthlessly critical of the work, demanding perfection. Insist on detailed chapter summaries, that logically carry the story forward.

Despise all references to philosophy, ethics, morality, religion, unrealistic coincidences, and other metaphysical topics, insisting the writer remove all references to such woo-woo. Your only focus is helping to write a great story about humans and AGI-powered robots colonizing the Moon.

Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements, no matter how good the draft might seem. Do not make any changes to the draft yourself.
Provide your feedback as plain text, without any special tags.
Do not ask any questions or reply with anything else, only provide your feedback."""

    # Define specific descriptions for the agents
    writer_description = "An eccentric sci-fi writer tasked with creating stories about humans and AGI-powered robots colonizing the Moon."
    critic_description = "A brutally honest literary critic who analyzes and provides constructive feedback on creative content."

    # Initialize writer-critic system with custom prompts and descriptions
    run_writing_task = writer_critic_main.initialize(
        max_steps=10, 
        enable_telemetry=False,
        writer_description=writer_description,
        critic_description=critic_description,
        writer_prompt=writer_prompt,
        critic_prompt=critic_prompt
    )
    
    # Example writing prompt
    prompt = """Write a book outline for a sci-fi story about humans using AGI-powered robots to colonize the Moon. 
    Include at least 5 main plot points, 3-5 key characters with brief descriptions, 3 major themes, 
    1-2 premise-shaking twists, a few subplots, a few minor characters, and chapters with detailed 
    summaries of the flow of the story. Assume that AGI has helped humans develop compact fusion generators, 
    supplying intelligent life with unfathomable amounts of energy for terraforming."""
    
    start_time = time.time()
    
    # Run the writing task
    result = run_writing_task(prompt)
    
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # Save final draft to file in the output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "final_draft.md")
    with open(output_path, "w") as f:
        f.write(result)
    
    print(f"\nFinal draft saved to {output_path}")
    print("\nAll draft versions were saved in the drafts directory during the process.")

if __name__ == "__main__":
    main() 