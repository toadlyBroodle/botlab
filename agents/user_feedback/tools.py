import os
import subprocess
import re
import time
import mailbox
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from smolagents import tool

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Constants
DEFAULT_MAILBOX_PATH = "/var/mail"
COMMAND_PATTERNS = {
    "frequency": r"FREQUENCY:\s*(\d+)",
    "detail": r"DETAIL:\s*(low|medium|high)",
    "focus": r"FOCUS:\s*(\w+)",
    "feedback": r"FEEDBACK:\s*(.*?)(?=\n\w+:|$)",
    "pause": r"PAUSE:\s*(true|false)",
    "resume": r"RESUME:\s*(true|false)"
}

@tool
def send_mail(recipient: str, subject: str, body: str) -> str:
    """Send an email using the system mail command.
    
    Args:
        recipient: Email address of the recipient
        subject: Subject line of the email
        body: Body content of the email
        
    Returns:
        Status message indicating success or failure
    """
    try:
        # Escape single quotes in the body and subject
        body_escaped = body.replace("'", "'\\''")
        subject_escaped = subject.replace("'", "'\\''")
        
        # Construct the command
        cmd = f"echo '{body_escaped}' | mail -s '{subject_escaped}' {recipient}"
        
        # Execute the command
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        logger.info(f"Email sent to {recipient} with subject: {subject}")
        return f"Email sent successfully to {recipient}"
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to send email: {e.stderr}"
        logger.error(error_msg)
        return error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error sending email: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def check_mail(username: Optional[str] = None, mailbox_path: Optional[str] = None) -> str:
    """Check for new emails in the system mailbox.
    
    Args:
        username: Optional username to check mail for (defaults to current user)
        mailbox_path: Optional path to mailbox (defaults to /var/mail/username)
        
    Returns:
        JSON string containing new messages
    """
    try:
        # Determine username and mailbox path
        if not username:
            username = os.getenv("USER") or subprocess.getoutput("whoami").strip()
        
        # Use provided mailbox_path if available, otherwise construct default path
        if not mailbox_path:
            mailbox_path = os.path.join(DEFAULT_MAILBOX_PATH, username)
        
        # Check if mailbox exists
        if not os.path.exists(mailbox_path):
            return f"No mailbox found at {mailbox_path}"
        
        # Read the mailbox
        mbox = mailbox.mbox(mailbox_path)
        
        # Get messages from the last 24 hours
        current_time = time.time()
        one_day_ago = current_time - (24 * 60 * 60)
        
        recent_messages = []
        for key, message in mbox.items():
            # Get message date
            date_str = message.get('Date')
            if not date_str:
                continue
                
            # Try to parse the date
            try:
                # Simple parsing for common date formats
                date_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
                message_time = date_obj.timestamp()
            except ValueError:
                try:
                    # Try alternative format
                    date_obj = datetime.strptime(date_str, "%d %b %Y %H:%M:%S %z")
                    message_time = date_obj.timestamp()
                except ValueError:
                    # If we can't parse the date, assume it's recent
                    message_time = current_time
            
            # Check if message is recent
            if message_time >= one_day_ago:
                # Extract relevant fields
                from_addr = message.get('From', 'Unknown')
                subject = message.get('Subject', 'No Subject')
                
                # Get message body
                body = ""
                if message.is_multipart():
                    for part in message.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                            break
                else:
                    body = message.get_payload(decode=True).decode('utf-8', errors='replace')
                
                recent_messages.append({
                    "from": from_addr,
                    "subject": subject,
                    "date": date_str,
                    "body": body
                })
        
        if recent_messages:
            return f"Found {len(recent_messages)} recent messages: {recent_messages}"
        else:
            return "No recent messages found"
    
    except Exception as e:
        error_msg = f"Error checking mail: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def parse_commands(email_body: str) -> Dict[str, Any]:
    """Parse commands from an email body.
    
    Args:
        email_body: The body text of the email to parse
        
    Returns:
        Dictionary of commands and their values
    """
    commands = {}
    
    # Extract commands using regex patterns
    for command, pattern in COMMAND_PATTERNS.items():
        match = re.search(pattern, email_body, re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            
            # Convert to appropriate types
            if command == "frequency":
                try:
                    value = int(value)
                except ValueError:
                    value = 1  # Default to 1 if conversion fails
            elif command in ["pause", "resume"]:
                value = value.lower() == "true"
                
            commands[command] = value
    
    if commands:
        return commands
    else:
        return {"status": "No commands found in email"} 