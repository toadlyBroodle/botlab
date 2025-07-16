from smolagents import CodeAgent
from ..utils.agents.tools import send_mail, check_mail, parse_email_commands, DEFAULT_MAILBOX_PATH
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
    """A class that handles automated user communication via email, including checking for commands and sending progress reports."""
    
    def __init__(
        self,
        model: Optional[RateLimitedLiteLLMModel] = None,
        max_steps: int = 1, # Max steps for report generation agent
        user_email: Optional[str] = None,
        report_frequency: int = 1,
        agent_description: Optional[str] = None, # Description for the report generating agent
        agent_prompt: Optional[str] = None, # System prompt for the report generating agent
        model_id: str = "gemini/gemini-2.0-flash",
        model_info_path: str = "agents/utils/gemini/gem_llm_info.json",
        base_wait_time: float = 2.0,
        max_retries: int = 3
    ):
        """Initialize the user feedback agent.
        
        Args:
            model: Optional RateLimitedLiteLLMModel to use. If not provided, one will be created.
            max_steps: Maximum number of steps for the internal report-generating agent.
            user_email: Email address to override the REMOTE_USER_EMAIL environment variable.
            report_frequency: How often to send reports (1 = every iteration).
            agent_description: Optional additional description for the report-generating agent.
            agent_prompt: Optional custom system prompt for the report-generating agent.
            model_id: The model ID to use if creating a new model.
            model_info_path: Path to the model info JSON file if creating a new model.
            base_wait_time: Base wait time for rate limiting if creating a new model.
            max_retries: Maximum retries for rate limiting if creating a new model.
        """
        if model is None:
            self.model = RateLimitedLiteLLMModel(
                model_id=model_id,
                model_info_path=model_info_path,
                base_wait_time=base_wait_time,
                max_retries=max_retries,
            )
        else:
            self.model = model
            
        self.max_steps = max_steps # For the internal agent
        
        self.local_email = os.getenv("LOCAL_USER_EMAIL", "fb_agent@botlab.dev")
        self.remote_email = user_email or os.getenv("REMOTE_USER_EMAIL")
        
        self.report_frequency = report_frequency
        self.iteration_count = 0
        self.maildir_path = DEFAULT_MAILBOX_PATH # Used for initial access checks
        
        logger.info(f"UserFeedbackAgent initialized with:")
        logger.info(f"- LOCAL_USER_EMAIL (sending FROM): {self.local_email}")
        logger.info(f"- REMOTE_USER_EMAIL (sending TO, checking FROM): {self.remote_email}")
        logger.info(f"- Maildir path for checks: {self.maildir_path}")
        
        if not self.local_email:
            logger.warning("LOCAL_USER_EMAIL environment variable is not set. Using default fb_agent@botlab.dev.")
        
        if not self.remote_email:
            logger.warning("REMOTE_USER_EMAIL environment variable is not set and no user_email provided. Email sending and checking will not work.")
        
        if user_email and user_email != os.getenv("REMOTE_USER_EMAIL"):
            os.environ["REMOTE_USER_EMAIL"] = user_email
            logger.info(f"Overriding REMOTE_USER_EMAIL with provided value: {user_email}")
        
        # Perform initial maildir access checks
        self.has_read_access = False
        self.can_mark_as_read = False # Determined by check_mail's ability to move files
        new_mail_dir = os.path.join(self.maildir_path, "new")
        cur_mail_dir = os.path.join(self.maildir_path, "cur")

        if os.path.exists(new_mail_dir) and os.path.isdir(new_mail_dir):
            self.has_read_access = os.access(new_mail_dir, os.R_OK)
            if os.path.exists(cur_mail_dir) and os.path.isdir(cur_mail_dir):
                 self.can_mark_as_read = os.access(new_mail_dir, os.W_OK) and os.access(cur_mail_dir, os.W_OK)

            if not self.has_read_access:
                logger.warning(f"No read access to {new_mail_dir}. Email checking will not work.")
            if not self.can_mark_as_read:
                 logger.warning(f"No write access to maildir ({new_mail_dir} or {cur_mail_dir}). Emails can be read but may not be marked as read by check_mail.")
        else:
            logger.warning(f"Maildir {new_mail_dir} doesn't exist. Email functionality will be limited.")
        
        # Create the internal agent for report generation ONLY
        base_description = "This agent is an expert at generating concise and informative progress reports based on provided state information."
        
        current_agent_description = agent_description if agent_description else base_description
            
        self._report_generator_agent = CodeAgent(
            tools=[], # No tools for the report generator
            model=self.model,
            name='report_generator_agent',
            description=current_agent_description,
            max_steps=self.max_steps, # Typically 1 for just generation
            additional_authorized_imports=[] # No complex imports needed for report generation
        )
        
        if agent_prompt:
            self._report_generator_agent.system_prompt = agent_prompt
        else:
            # Apply default smolagent prompts if no custom one is given
            # These might need adjustment if they imply tool use.
            # For now, we assume agent_prompt or a simple default is sufficient.
            try:
                apply_custom_agent_prompts(self._report_generator_agent) 
            except Exception as e:
                logger.warning(f"Could not apply default smolagent prompts to report generator: {e}. Using basic system prompt.")
                self._report_generator_agent.system_prompt = "You generate progress reports."

    @property
    def agent(self):
        """Get the underlying report-generating CodeAgent instance."""
        return self._report_generator_agent
    
    def process_feedback(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks for user email commands, updates state, generates a progress report,
        and sends it to the user.
        
        Args:
            state: The current state of the agent loop.
            
        Returns:
            Updated state with user feedback incorporated.
        """
        self.iteration_count += 1
        logger.info(f"UserFeedbackAgent - Iteration {self.iteration_count}")

        if "user_commands" not in state:
            state["user_commands"] = {}
        
        # 1. Check for new emails and parse commands
        if self.has_read_access and self.remote_email:
            logger.info(f"Checking for emails from {self.remote_email}...")
            email_data = check_mail() # This function now handles its own logging for found/not found
            
            if email_data and isinstance(email_data, dict) and "error" not in email_data:
                if email_data.get("body"):
                    logger.info(f"Found email with subject: '{email_data.get('subject', 'No Subject')}' from {email_data.get('from')}. Parsing commands...")
                    commands = parse_email_commands(email_data["body"])
                    
                    if commands and isinstance(commands, dict) and "status" not in commands:
                        logger.info(f"Parsed commands: {commands}")
                        state["user_commands"].update(commands)
                        
                        # Handle specific commands directly
                        if "frequency" in commands and isinstance(commands["frequency"], int):
                            self.report_frequency = commands["frequency"]
                            logger.info(f"Report frequency updated to: {self.report_frequency}")
                        if "pause" in commands:
                            state["paused"] = commands["pause"]
                            logger.info(f"Pause state updated to: {state['paused']}")
                        if "resume" in commands: # resume is effectively pause = False
                            state["paused"] = not commands["resume"]
                            logger.info(f"Pause state updated (via resume) to: {state['paused']}")
                        # Other commands like 'detail', 'focus', 'feedback' are stored in state['user_commands']
                        # for other agents to potentially use.
                    elif commands and "status" in commands:
                        logger.info(f"Command parsing status: {commands['status']}")
                    else:
                        logger.info("No commands found in email body or empty commands dict returned.")
                else:
                    logger.info("No new email body found from relevant sender.")
            elif email_data and "error" in email_data:
                logger.error(f"Error checking mail: {email_data['error']}")
            # If email_data is empty dict, check_mail already logged "No new mail files found" or "No relevant emails"
            
        else:
            if not self.remote_email:
                logger.info("Email checking and sending skipped: REMOTE_USER_EMAIL not configured.")
            elif not self.has_read_access:
                logger.info("Email checking skipped: No read access to maildir.")

        # 2. Determine if a progress report should be sent
        is_paused = state.get("paused", False)
        if self.should_report() and self.remote_email and not is_paused:
            logger.info(f"Generating and sending progress report for iteration {self.iteration_count}.")
            
            # 3. Generate report content using the internal LLM agent
            report_content_body = self.generate_report_content(state)
            
            if report_content_body:
                # 4. Send the report
                subject = f"Agent Progress Report - Iteration {self.iteration_count}"
                send_status = send_mail(subject, report_content_body)
                logger.info(f"Report sending status to {self.remote_email}: {send_status}")
                # Save the generated report for record-keeping if needed using daily master files
                save_final_answer(self._report_generator_agent, report_content_body, f"Iteration {self.iteration_count} report to user", "user_feedback_report", use_daily_master=True)
            else:
                logger.warning("Report content generation failed or returned empty. Report not sent.")
        elif is_paused:
            logger.info(f"Report sending skipped for iteration {self.iteration_count}: System is PAUSED.")
        elif not self.remote_email:
            logger.info(f"Report sending skipped: REMOTE_USER_EMAIL not configured.")

        # For general logging of this cycle's outcome, independent of LLM result
        # save_final_answer(self._report_generator_agent, f"Process feedback cycle {self.iteration_count} complete.", "user_feedback_cycle_log", use_daily_master=True)
        # Decided to save only actual reports for now.

        return state
    
    def should_report(self) -> bool:
        """Determine if a report should be sent based on the current iteration and frequency."""
        return self.iteration_count % self.report_frequency == 0
    
    def generate_report_content(self, state: Dict[str, Any]) -> Optional[str]:
        """Generate a concise progress report body using the internal LLM agent.
        
        Args:
            state: The current state of the agent loop.
            
        Returns:
            The generated report string, or None if generation fails.
        """
        if not self.remote_email: # Should be caught by caller, but good to double check
            logger.warning("Cannot generate report content: No external email address configured.")
            return None
        
        logger.info(f"Generating report content for iteration {self.iteration_count} to be sent to {self.remote_email}")
        
        # Construct a prompt for the report-generating agent
        # The system prompt of _report_generator_agent already guides its role.
        # Here, we provide the context (current state).
        prompt = f"""
        Based on the following current state of the automated process, please generate a concise and informative progress report body.
        The report will be sent to a user. Focus on high-level achievements, significant changes, any issues encountered, and planned next steps.
        Keep it brief, like a short email update. Do not include salutations like "Dear User" or closings like "Sincerely", just the body.

        Current Iteration: {self.iteration_count}
        Overall Query/Goal: {state.get("query", "N/A")}
        Current Agent/Task: {state.get("current_agent", "N/A")}
        Progress Details:
        {state.get("loop_results", {})}
        
        Any User Commands Received This Cycle:
        {state.get("user_commands", {})}

        Current System Status (e.g., paused):
        Paused: {state.get("paused", False)}
        
        Please provide only the report body text.
        """
        
        try:
            report_body = self._report_generator_agent(prompt)
            if report_body and isinstance(report_body, str) and report_body.strip():
                logger.info(f"Report content generated successfully, length: {len(report_body)} chars.")
                return report_body.strip()
            else:
                logger.warning(f"Report generator agent returned empty or invalid content: {report_body}")
                return None
        except Exception as e:
            logger.error(f"Error during report content generation by LLM agent: {str(e)}")
            return None 