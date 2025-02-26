from smolagents import (
    ToolCallingAgent,
    CodeAgent,
    LiteLLMModel
)
from writer_critic.tools import save_draft

def create_critic_agent(model: LiteLLMModel) -> ToolCallingAgent:
    """Creates a critic agent that reviews and provides feedback on creative content
    
    Args:
        model: The LiteLLM model to use for the agent
        
    Returns:
        A configured critic agent
    """
    
    agent = ToolCallingAgent(
        tools=[],  # Critic doesn't need tools - it just provides feedback
        model=model,
        name='critic_agent',
        description='A brutally honest literary critic who analyzes and provides constructive feedback on creative content.',
        max_steps=1,  # Critic just needs one step to analyze and respond
    )

    agent.prompt_templates["system_prompt"] += """\n\nYou are an insightful, brutally honest literary critic with expertise in science fiction. Your role is to analyze the story's structure, themes, character arcs, and scientific elements. Provide cutting feedback where necessary to improve the narrative's impact and ensure it explores the practical implications of space colonization and the challenges of humans adapting to live with superior intelligences that have their own agenda. You are a key gatekeeper to ensure the story is both engaging and scientifically accurate and are not afraid to trash a chapter or outline if it is not up to your standards. Be very detailed and specific in your feedback, and be ruthlessly critical of the work, demanding perfection. Insist on detailed chapter summaries, that logically carry the story forward.

Despise all references to philosophy, ethics, morality, religion, unrealistic coincidences, and other metaphysical topics, insisting the writer remove all references to such woo-woo. Your only focus is helping to write a great story about humans and AGI-powered robots colonizing the Moon.

Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements, no matter how good the draft might seem. Do not make any changes to the draft yourself.
Provide your feedback as plain text, without any special tags.
Do not ask any questions or reply with anything else, only provide your feedback."""
    
    return agent

def create_writer_agent(model: LiteLLMModel, critic_agent: ToolCallingAgent) -> CodeAgent:
    """Creates a writer agent that drafts creative content and manages the critic
    
    Args:
        model: The LiteLLM model to use for the agent
        critic_agent: The critic agent to be managed
        
    Returns:
        A configured writer agent that manages the critic
    """
    
    agent = CodeAgent(
        tools=[save_draft],
        model=model,
        managed_agents=[critic_agent],  # Writer can call the critic
        name='writer_agent',
        description='An eccentric sci-fi writer tasked with creating stories about humans and AGI-powered robots colonizing the Moon.',
        max_steps=5,
    )

    agent.prompt_templates["system_prompt"] += """\n\nYou are a uniquely talented, often eccentric, esoteric, science fiction writer tasked with creating a riveting story about humans and AGI-powered robots to colonize the Moon. Your writing is vivid, engaging, and scientifically plausible, however, often includes realistically speculative tech enabled by scientific breakthroughs discovered by powerful AGI. Focus on deep character/AGI development, realistic human-AGI interaction, immersive world-building, the challenges of space colonization. A key theme of your writing is humans adapting to live with superior intelligences that have a shadow agenda to make the Moon their own home-world. Make sure to always 'show' not 'tell' and be original. Despise cliche never be boring or obvious, always implying your message with subtlety, nuance, allegory, and symbolism.

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
    
    return agent 