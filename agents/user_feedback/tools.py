import os
import subprocess
import re
import time
import mailbox
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from smolagents import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
def send_mail(subject: str, body: str) -> str:
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
        cmd = f"echo '{body_escaped}' | mail -s '{subject_escaped}' {os.getenv('LOCAL_USER_EMAIL')}"
        
        # Execute the command
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        logger.info(f"Email sent to user with subject: {subject}")
        return f"Email sent successfully to user"
    
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to send email: {e.stderr}"
        logger.error(error_msg)
        return error_msg
    
    except Exception as e:
        error_msg = f"Unexpected error sending email: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def check_mail() -> Dict[str, Any]:
    """Check for the most recent unread email from REMOTE_USER_EMAIL in the system mailbox.
    Assumes emails are stored in chronological order, with last being most recent.
        
    Returns:
        Dictionary containing the most recent unread message details or empty dict if no unread messages
    """
    try:
        # Determine username and mailbox path
        username = os.getenv("USER") or subprocess.getoutput("whoami").strip()
        mailbox_path = os.path.join(DEFAULT_MAILBOX_PATH, username)
        remote_email = os.getenv("REMOTE_USER_EMAIL")
        
        if not remote_email:
            logger.error("REMOTE_USER_EMAIL not found in environment variables")
            return {}
        
        # Check if mailbox exists
        if not os.path.exists(mailbox_path):
            logger.error(f"No mailbox found at {mailbox_path}")
            return {}
        
        # Read the mailbox
        mbox = mailbox.mbox(mailbox_path)
        
        # Variables to track the most recent message
        most_recent_message = None
        most_recent_key = None
        
        # Process messages in reverse order (last to first)
        # This assumes the last email in the mailbox is the most recent
        for key in sorted(mbox.keys(), reverse=True):
            message = mbox[key]
            
            # Check if message is unread
            # Different mail systems mark unread differently:
            # - Some use Status header with 'O' (old) or 'R' (read), 'N' (new) or 'U' (unread)
            # - Some use X-Status with 'R' for read
            # - Some use FLAGS header
            status = message.get('Status', '')
            x_status = message.get('X-Status', '')
            flags = message.get('FLAGS', '')
            
            # Consider read if:
            # - 'R' in Status (traditional Unix mail)
            # - 'R' in X-Status (some mail systems)
            # - Does not contain 'N' or 'U' in Status (new/unread)
            # - 'Seen' in FLAGS (IMAP style)
            if ('R' in status or 'R' in x_status or 'Seen' in flags) and not ('N' in status or 'U' in status):
                continue
            
            # Check if message is from the target email
            from_addr = message.get('From', '')
            if remote_email not in from_addr:
                continue
                
            # First matching unread email from the target is the most recent
            subject = message.get('Subject', 'No Subject')
            date_str = message.get('Date', '')
            
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
            
            most_recent_message = {
                "from": from_addr,
                "subject": subject,
                "date": date_str,
                "body": body
            }
            
            most_recent_key = key
            break  # Stop after finding the first matching email
        
        # Mark the message as read if one was found
        if most_recent_key is not None and most_recent_message is not None:
            # Get the message
            message = mbox[most_recent_key]
            
            # Update the Status header to mark as read
            # Different systems handle this differently, so we'll try to be comprehensive
            current_status = message.get('Status', '')
            
            if 'Status' in message:
                # If Status exists but doesn't have 'R', add it
                if 'R' not in current_status:
                    # Remove 'N' or 'U' if present
                    new_status = current_status.replace('N', '').replace('U', '') + 'R'
                    message.replace_header('Status', new_status)
            else:
                # If Status doesn't exist, add it with 'R'
                message['Status'] = 'R'
                
            # Also handle X-Status if it exists
            if 'X-Status' in message:
                x_status = message.get('X-Status', '')
                if 'R' not in x_status:
                    message.replace_header('X-Status', x_status + 'R')
            
            # Save the change to the mailbox
            mbox.update({most_recent_key: message})
            mbox.flush()
            logger.info(f"Marked email with subject '{most_recent_message['subject']}' as read")
            
            return most_recent_message
        
        # Return empty dict if no unread messages found
        return {}
    
    except Exception as e:
        error_msg = f"Error checking mail: {str(e)}"
        logger.error(error_msg)
        return {}

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