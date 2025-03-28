from smolagents import CodeAgent
from .tools import send_mail, check_mail, parse_commands, FB_AGENT_USER
from ..utils.agents.tools import save_final_answer, apply_custom_agent_prompts
from ..utils.gemini.rate_lim_llm import RateLimitedLiteLLMModel
from typing import Optional, Dict, Any, List
import os
import time


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
            user_email: Email address of the user to communicate with
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
        self.user_email = user_email or os.getenv("LOCAL_USER_EMAIL")
        self.report_frequency = report_frequency
        self.iteration_count = 0
        self.feedback_agent_user = FB_AGENT_USER
        
        # Check if environment variables are properly set up
        if not self.user_email:
            print("LOCAL_USER_EMAIL environment variable is not set. Email sending will not work.")
        
        if not os.getenv("REMOTE_USER_EMAIL"):
            print("REMOTE_USER_EMAIL environment variable is not set. Email checking will not work.")
        
        # Verify mailbox access
        mailbox_path = f"/var/mail/{self.feedback_agent_user}"
        self.has_read_access = False
        self.has_write_access = False
        
        if os.path.exists(mailbox_path):
            self.has_read_access = os.access(mailbox_path, os.R_OK)
            self.has_write_access = os.access(mailbox_path, os.W_OK)
            
            if not self.has_read_access:
                print(f"No read access to {mailbox_path}. Email checking will not work.")
            if not self.has_write_access:
                print(f"No write access to {mailbox_path}. Emails can be read but not marked as read.")
        else:
            print(f"Mailbox {mailbox_path} doesn't exist. Email functionality will be limited.")
        
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
        
        # Skip email checking if we don't have proper access
        email_instructions = ""
        if self.has_read_access:
            email_instructions = f"""
            1. Check for new emails and extract any commands or feedback
               - Use check_mail() to check for unread emails from REMOTE_USER_EMAIL in the {self.feedback_agent_user} mailbox
            2. Update the state based on user feedback if any
            """
        else:
            email_instructions = """
            Note: Email checking is disabled due to insufficient permissions.
            """
        
        # Check for new emails and process commands
        prompt = f"""
        You are the UserFeedbackAgent responsible for communicating with the user via email.
        
        Current iteration: {self.iteration_count}
        Report frequency: Every {self.report_frequency} iterations
        User email: {self.user_email}
        Feedback agent user: {self.feedback_agent_user}
        
        First, check for new emails from the user that might contain feedback or commands.
        Then, determine if you should send a progress report based on the current iteration and report frequency.
        
        Current state:
        {state}
        
        {email_instructions}
        3. Determine if a progress report should be sent
        4. If needed, generate and send a concise progress report using send_mail()
        """
        
        result = self._agent(prompt)
        
        # Extract user commands from the result
        if "user_commands" not in state:
            state["user_commands"] = {}
            
        # Look for command processing in the result
        if self.has_read_access and ("commands found" in result.lower() or "processed command" in result.lower()):
            # Check if we received any emails
            emails = check_mail()
            if emails and "body" in emails:
                # Parse commands from the email body
                commands = parse_commands(emails["body"])
                if commands and isinstance(commands, dict):
                    state["user_commands"].update(commands)
                    
                    # Log the commands we found
                    print(f"Extracted user commands: {commands}")
        
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
        prompt = f"""
        Generate a concise progress report for the user based on the current state.
        Focus on high-level achievements, changes, and next steps.
        Keep it brief but informative.
        
        Current state:
        {state}
        """
        
        result = self._agent(prompt)
        return result 