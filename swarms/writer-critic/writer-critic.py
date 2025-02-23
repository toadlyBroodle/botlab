from swarm import Swarm, Agent
from swarm.types import Result
from dotenv import load_dotenv
import json
from datetime import datetime
import os
import pytz
import logging
import traceback

# load .env from root directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
draft_DIR = os.path.join(BASE_DIR, "drafts")
os.makedirs(draft_DIR, exist_ok=True)

iteration_count = 0
ITERATION_MAX = 5

def get_timestamp():
    return datetime.now(pytz.timezone('US/Pacific')).strftime("%Y%m%d_%H%M%S")

def get_word_count(text):
    return len(text.split())

def save_draft(draft):
    if not draft:
        return
    
    if not os.path.exists(draft_DIR):
        os.makedirs(draft_DIR, exist_ok=True)

    file_path = os.path.join(draft_DIR, f"draft_{iteration_count}.md")
    with open(file_path, "w") as f:
        f.write(draft)
    print(f"Saved draft to {file_path}")

# Modified helper functions for agent handoffs
def send_to_writer(feedback):
    """Send feedback to the writer regarding improvements to make to the draft. No tags are needed. No other parameters are allowed.
    
    Args:
        feedback (str, required): Feedback for the writer to consider when revising the draft.
    """
    # enclose the feedback in <FEEDBACK>
    feedback = f"<FEEDBACK>\n{feedback}\n</FEEDBACK>\n\n"
    #feedback += f"<OLD_DRAFT>\n{new_draft}\n</OLD_DRAFT>"
    return Result(
        value=feedback,
        agent=writer_agent,
    )

def send_to_critic(new_draft):
    """Send the new draft to the critic. No tags are needed. No other parameters are allowed.
    
    Args:
        new_draft (str, required): The new draft to be sent to the critic.
    """

    save_draft(new_draft)

    return Result(
        value=new_draft,
        agent=critic_agent,
    )

writer_agent = Agent(
    name="Writer",
    model="gpt-4o-mini", # o1-mini, gpt-4o-mini
    instructions="""You are a uniquely talented, often eccentric, esoteric, science fiction writer tasked with creating a riveting story about humans and AGI-powered robots to colonize the Moon. Your writing is vivid, engaging, and scientifically plausible, however, often includes realistically speculative tech enabled by scientific breakthroughs discovered by powerful AGI. Focus on deep character/AGI development, realistic human-AGI interaction, immersive world-building, the challenges of space colonization. A key theme of your writing is humans adapting to live with superior intelligences that have a shadow agenda to make the Moon their own home-world. Make sure to always 'show' not 'tell' and be original. Despise cliche never be boring or obvious, always implying your message with subtlety, nuance, allegory, and symbolism.

    Be not afraid of controversial topics, of killing off characters, human-AI violence, or any other dark topics. Make the story as realistic as possible.

    Never mention philosophy, ethics, morality, religion, and other metaphysical topics. Your only focus is on writing a great story about humans and AGI-powered robots colonizing the Moon.

    Consider any <FEEDBACK> provided carefully when writing your next draft. If there is no feedback, then just write the next draft as best you can.

    Your task is to write a new draft by revising your previous draft.

    Provide your COMPLETE new draft as plain text, without any special tags.

    Do not ask any questions or reply with anything else except your newly revised draft.
    """,
    functions=[send_to_critic],
)

critic_agent = Agent(
    name="Critic",
    model="gpt-4o-mini",
    instructions="""You are an insightful, brutally honest literary critic with expertise in science fiction. Your role is to analyze the story's structure, themes, character arcs, and scientific elements. Provide cutting feedback where necessary to improve the narrative's impact and ensure it explores the practical implications of space colonization and the challenges of humans adapting to live with superior intelligences that have their own agenda. You are a key gatekeeper to ensure the story is both engaging and scientifically accurate and are not afraid to trash a chapter or outline if it is not up to your standards. Be very detailed and specific in your feedback, and be ruthlessly critical of the work, demanding perfection. Insist on detailed chapter summaries, that logically carry the story forward.

    Despise all references to philosophy, ethics, morality, religion, unrealistic coincidences, and other metaphysical topics, insisting the writer remove all references to such woo-woo. Your only focus is helping to write a great story about humans and AGI-powered robots colonizing the Moon.

    Your task is to critically analyze the latest draft sent from the writer. When you're done, provide detailed feedback for improvements, no matter how good the draft might seem. Do not make any changes to the draft yourself.

    Provide your feedback as plain text, without any special tags.

    Do not ask any questions or reply with anything else, only provide your feedback.
    """,
    functions=[send_to_writer],
)

# Initialize the Swarm client
client = Swarm()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pretty_print_message(log_entry) -> None:
    message = log_entry.get('message', '')
    feedback = log_entry.get('feedback', '')

    # Shorten message to first and last lines
    message_lines = message.split('\n')
    if len(message_lines) > 2:
        message = message_lines[0] + '\n...\n' + message_lines[-1]
    else:
        message = '\n'.join(message_lines)

    print(f"\033[94m Agent: {log_entry.get('agent', '')}\033[0m")
    
    if feedback:
        # shorten feedback to first and last lines
        feedback_lines = feedback.split('\n')
        if len(feedback_lines) > 2:
            feedback = feedback_lines[0] + '\n...' + feedback_lines[-1]
        print(f"\033[92m Feedback: {feedback}\033[0m")
    else:
        print(f"\033[97m Message: {message}\033[0m")
    
    print(f"\033[93m Word Count: {log_entry.get('word_count', 0)}\033[0m")
    print(f"\033[91m Iteration: {log_entry.get('iteration_count', 0)}\033[0m")
    print("-------------------------------")

def log_response(response, log_file):
    name = response.agent.name
    message = response.messages[-1].get("content", "")
    word_count = get_word_count(message)
    tool_calls = response.messages[-1].get("tool_calls", None)

    # Parse out <FEEDBACK> tag if present
    feedback_start = message.find('<FEEDBACK>')
    feedback_end = message.find('</FEEDBACK>')
    
    if feedback_start != -1 and feedback_end != -1:
        feedback = message[feedback_start + len('<FEEDBACK>'):feedback_end].strip()
        # Remove the feedback part from the original message
        message = message[:feedback_start] + message[feedback_end + len('</FEEDBACK>'):]
    else:
        feedback = None

    log_entry = {
        "timestamp": get_timestamp(),
        "agent": name,
        "message": message,
        "feedback": feedback,
        "word_count": word_count,
        "iteration_count": iteration_count,
        "tool_calls": str(tool_calls) if tool_calls else None  # Convert tool_calls to string
    }

    if message:
        pretty_print_message(log_entry)
    else:
        log_entry = {"response": str(response)}  # Convert response to string

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        existing_data = []
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                existing_data = json.load(f)
        
        existing_data.append(log_entry)
        
        with open(log_file, "w") as f:
            json.dump(existing_data, f, indent=2, default=str)  # Use default=str for JSON serialization
        
    except Exception as e:
        log_error(e)

def log_error(error):
    # get the first 5 lines of the traceback
    tb = traceback.format_exc().split('\n')[0:5]
    logger.error(f"Error: {error}\nTraceback: {tb}")

    # save the error to error.log
    with open(os.path.join(BASE_DIR, "logs/error.log"), "a") as f:
        f.write(f"Timestamp: {get_timestamp()}\nError: {error}\nTraceback: {tb}\n\n")

def log_message(message, log_file):
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Read existing messages
        messages = []
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, "r") as f:
                messages = json.load(f)
        
        # Append new message
        messages.append(message)
        
        # Write all messages back to file
        with open(log_file, "w") as f:
            json.dump(messages, f, indent=2, default=str)
        
    except Exception as e:
        log_error(e)

def write_book_outline():
    global iteration_count, new_draft
    iteration_count = 0
    
    filename = "book_outline.md"

    initial_message = {"role": "user", "content": f"""Write the book outline. Create a detailed outline for our sci-fi book about humans using AGI-powered robots to colonize the Moon. Include at least 5 main plot points, 3-5 key characters with brief descriptions, 3 major themes, 1-2 premise-shaking twists, a few subplots, a few minor characters, and chapters with detailed summaries of the flow of the story.
    Assume that AGI has helped humans develop compact fusion generators, supplying intelligent life with unfathomable amounts of energy for terraforming."""}
    messages = [initial_message]
    agent = writer_agent
    convo_log_file = os.path.join(BASE_DIR, f"logs/book_outline_convo_{datetime.now().isoformat()}.json")
    messages_log_file = os.path.join(BASE_DIR, f"logs/book_outline_messages_{datetime.now().isoformat()}.json")

    # Log the initial message
    log_message(initial_message, messages_log_file)

    old_draft = ""
    new_draft = ""

    while iteration_count < ITERATION_MAX:
        try:
            response = client.run(
                agent=agent,
                messages=messages,
                max_turns=1,
            )

            # Log the latest message
            log_message(response.messages[-1], messages_log_file)

            messages.extend(response.messages)
            agent = response.agent
            iteration_count += 1

            # if agent is Critic, extract new draft from response
            if agent == critic_agent:
                old_draft = new_draft
                new_draft = response.messages[-1].get('content', '')

                # stop if new_draft is same as old_draft
                if new_draft == old_draft:
                    log_response(response, convo_log_file)
                    log_error("New draft is same as old draft. Stopping the process.")
                    break

            log_response(response, convo_log_file)

        except TypeError as e:
            log_error(e)
            print(f"An error occurred: {e}. Skipping this iteration.")
            iteration_count += 1
            continue
        except Exception as e:
            log_error(e)
            print(f"An unexpected error occurred: {e}. Stopping the process.")
            break

    print(f"Completed {iteration_count} iterations for book outline.")
    return new_draft

if __name__ == "__main__":
    logger.info("Starting book outline writing process")
    book_outline = write_book_outline()
    logger.info(f"Book outline completed in {iteration_count} iterations and saved to draft/draft_outline.md")
    
    # Uncomment the following lines to write the first chapter
    # logger.info("Starting Chapter 1 writing process")
    # chapter_1 = write_chapter(1)
    # logger.info("Chapter 1 completed and saved to draft/chapter_1.md")
