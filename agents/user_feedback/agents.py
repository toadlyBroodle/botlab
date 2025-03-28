from smolagents import CodeAgent
from .tools import send_mail, check_mail, parse_commands, FB_AGENT_USER, DEFAULT_MAILBOX_PATH
from ..utils.agents.tools import save_final_answer, apply_custom_agent_prompts
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from typing import Optional, Dict, Any, List
import os
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class UserFeedbackAgent:
    """A wrapper class for the user feedback agent that handles communication with users via email."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 5,
        user_email: Optional[str] = None,
        report_frequency: int = 1,
        agent_description: Optional[str] = None,
        agent_prompt: Optional[str] = None,
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the user feedback agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the agent
            user_email: Email address to override the REMOTE_USER_EMAIL environment variable
            report_frequency: How often to send reports (1 = every iteration)
            agent_description: Optional additional description to append to the base description
            agent_prompt: Optional custom system prompt to use instead of the default
            model_id: The model ID to use if creating a new model
            model_info_path: Path to the model info JSON file if creating a new model
            base_wait_time: Base wait time for rate limiting if creating a new model
            max_retries: Maximum retries for rate limiting if creating a new model
        """
        # Create a model if one wasn't provided
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        self.max_steps = max_steps
        
        # LOCAL_USER_EMAIL is the email of the system user (used in mailbox)
        # REMOTE_USER_EMAIL is the external user's email for sending/receiving
        self.local_email = os.getenv("LOCAL_USER_EMAIL", "fb_agent@botlab.dev")
        self.remote_email = user_email or os.getenv("REMOTE_USER_EMAIL")
        
        self.report_frequency = report_frequency
        self.iteration_count = 0
        self.feedback_agent_user = FB_AGENT_USER
        self.maildir_path = DEFAULT_MAILBOX_PATH
        
        # Log important configuration
        logger.info(f"UserFeedbackAgent initialized with:")
        logger.info(f"- LOCAL_USER_EMAIL: {self.local_email}")
        logger.info(f"- REMOTE_USER_EMAIL: {self.remote_email}")
        logger.info(f"- Maildir path: {self.maildir_path}")
        
        # Check if environment variables are properly set up
        if not self.local_email:
            logger.warning("LOCAL_USER_EMAIL environment variable is not set. Using default fb_agent@botlab.dev.")
        
        if not self.remote_email:
            logger.warning("REMOTE_USER_EMAIL environment variable is not set and no user_email provided. Email sending and checking will not work.")
            
        # Override environment variable if user_email was provided
        if user_email and user_email != os.getenv("REMOTE_USER_EMAIL"):
            # Temporarily set the environment variable for tools to use
            os.environ["REMOTE_USER_EMAIL"] = user_email
            logger.info(f"Overriding REMOTE_USER_EMAIL with provided value: {user_email}")
        
        # Verify mailbox access
        new_mail_dir = os.path.join(self.maildir_path, "new")
        self.has_read_access = False
        self.has_write_access = False
        
        if os.path.exists(new_mail_dir):
            self.has_read_access = os.access(new_mail_dir, os.R_OK)
            self.has_write_access = os.access(new_mail_dir, os.W_OK)
            
            if not self.has_read_access:
                logger.warning(f"No read access to {new_mail_dir}. Email checking will not work.")
            if not self.has_write_access:
                logger.warning(f"No write access to {new_mail_dir}. Emails can be read but not marked as read.")
        else:
            logger.warning(f"Maildir {new_mail_dir} doesn't exist. Email functionality will be limited.")
        
        # Create the agent
        base_description = f"""This agent handles communication with users via email using the dedicated {self.feedback_agent_user} user. It can check for new emails from users, parse commands and feedback, and send progress reports and updates to users. Use this agent to maintain communication with users during long-running agent processes."""
        
        # Append additional description if provided
        if agent_description:
            description = f"{base_description} {agent_description}"
        else:
            description = base_description
            
        # Create the agent with either the default or custom prompt
        self._agent = CodeAgent(
            tools=[send_mail, check_mail, parse_commands],
            model=self.model,
            name='user_feedback_agent',
            description=description,
            max_steps=self.max_steps,
            additional_authorized_imports=["json", "time", "re"]
        )
        
        # Set custom prompt if provided
        if agent_prompt:
            self._agent.system_prompt = agent_prompt
    
    @property
    def agent(self):
        """Get the underlying CodeAgent instance."""
        return self._agent
    
    def process_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback and update the state accordingly.
        
        Args:
            state: The current state of the agent loop
            
        Returns:
            Updated state with user feedback incorporated
        """
        self.iteration_count += 1
        
        # Create an explanation of the email setup for the agent
        email_explanation = f"""
        Email Configuration:
        - The agent's local email address is: {self.local_email}
        - Reports are sent TO: {self.remote_email or 'Not configured'}
        - Replies are expected FROM: {self.remote_email or 'Not configured'}
        - Emails are stored in Maildir format at: {self.maildir_path}
        """
        
        # Skip email checking if we don't have proper access
        email_instructions = ""
        if self.has_read_access and self.remote_email:
            email_instructions = f"""
            1. First, check for new emails containing feedback or commands:
               - Use check_mail() to check for unread emails from {self.remote_email}
               - New emails will be in the 'new' folder of the maildir: {os.path.join(self.maildir_path, 'new')}
               - If you find an email, parse any commands using parse_commands() on the email body
            2. Update the state based on user feedback if any
            """
        else:
            if not self.has_read_access:
                email_instructions = """
                Note: Email checking is disabled due to insufficient maildir permissions.
                """
            elif not self.remote_email:
                email_instructions = """
                Note: Email checking is disabled because REMOTE_USER_EMAIL is not set.
                """
            else:
                email_instructions = """
                Note: Email checking is disabled due to configuration issues.
                """
        
        # Check for new emails and process commands
        prompt = f"""
        You are the UserFeedbackAgent responsible for communicating with the user via email.
        
        {email_explanation}
        
        Current iteration: {self.iteration_count}
        Report frequency: Every {self.report_frequency} iterations
        
        First, check for new emails from the user that might contain feedback or commands.
        Then, determine if you should send a progress report based on the current iteration and report frequency.
        
        Current state:
        {state}
        
        {email_instructions}
        3. Determine if a progress report should be sent
        4. If needed, generate and send a concise progress report using send_mail()
           - The report should be sent TO: {self.remote_email}
           - Be sure to include a descriptive subject line
        """
        
        result = self._agent(prompt)
        logger.info(f"UserFeedbackAgent processed feedback for iteration {self.iteration_count}")
        
        # Extract user commands from the result
        if "user_commands" not in state:
            state["user_commands"] = {}
            
        # Look for command processing in the result
        if self.has_read_access and self.remote_email and ("commands found" in result.lower() or "processed command" in result.lower()):
            # Check if we received any emails
            logger.info("Detected possible commands, checking mail directly")
            emails = check_mail()
            if emails and "body" in emails:
                # Parse commands from the email body
                logger.info(f"Found email with subject: {emails.get('subject', 'No subject')}")
                commands = parse_commands(emails["body"])
                if commands and isinstance(commands, dict) and not "status" in commands:
                    state["user_commands"].update(commands)
                    
                    # Log the commands we found
                    logger.info(f"Extracted user commands: {commands}")
                    print(f"Extracted user commands: {commands}")
                else:
                    logger.info("No commands found in email body or only status returned")
            else:
                logger.info("No relevant emails found or error checking mail")
        
        # Save the result to the agent's data directory
        save_final_answer(self._agent, result, "user_feedback")
        
        return state
    
    def should_report(self) -> bool:
        """Determine if a report should be sent based on the current iteration and frequency."""
        return self.iteration_count % self.report_frequency == 0
    
    def generate_report(self, state: Dict[str, Any]) -> str:
        """Generate a concise progress report based on the current state.
        
        Args:
            state: The current state of the agent loop
            
        Returns:
            A formatted progress report
        """
        if not self.remote_email:
            logger.warning("Cannot generate report: No external email address configured")
            return "Cannot generate report: No external email address configured. Set REMOTE_USER_EMAIL environment variable."
        
        logger.info(f"Generating report for iteration {self.iteration_count}")
        
        prompt = f"""
        Generate a concise progress report for the user based on the current state.
        Focus on high-level achievements, changes, and next steps.
        Keep it brief but informative.
        
        Email will be sent FROM: {self.local_email}
        Email will be sent TO: {self.remote_email}
        
        The subject line should include the progress status and iteration number.
        
        Current state:
        {state}
        """
        
        result = self._agent(prompt)
        
        # Log the report generation
        logger.info(f"Generated report for {self.remote_email}, length: {len(result)} chars")
        
        # Check if send_mail was called in the result
        if "Email sent successfully" not in result and "successfully sent" not in result.lower():
            logger.warning("Report generated but no indication that email was sent. Sending directly.")
            email_subject = f"Agent Progress Report - Iteration {self.iteration_count}"
            send_result = send_mail(email_subject, result)
            logger.info(f"Manual email sending result: {send_result}")
        
        return result 