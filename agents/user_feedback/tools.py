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
DEFAULT_MAILBOX_PATH = "/home/fb_agent/var/mail"
FB_AGENT_USER = "fb_agent"  # Dedicated user for feedback
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
    """Send an email using the sendmail command with explicit headers.
    
    Args:
        subject: Subject line of the email
        body: Body content of the email
        
    Returns:
        Status message indicating success or failure
    """
    try:
        # Get the recipient email from environment
        recipient = os.getenv("REMOTE_USER_EMAIL")
        sender = os.getenv("LOCAL_USER_EMAIL") or "fb_agent@botlab.dev"
        
        if not recipient:
            error_msg = "REMOTE_USER_EMAIL environment variable is not set. Cannot send email."
            logger.error(error_msg)
            return error_msg
        
        # Escape special characters that might cause issues with the shell command
        subject_escaped = subject.replace('"', '\\"').replace("'", "\\'").replace('`', '\\`')
        body_escaped = body.replace('"', '\\"').replace("'", "\\'").replace('`', '\\`')
        
        # Create a timestamp for the Date header
        timestamp = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
        
        # Create email with explicit headers
        email_content = f'From: {sender}\nTo: {recipient}\nSubject: {subject_escaped}\nDate: {timestamp}\n\n{body_escaped}'
        
        # Use sendmail with explicit headers
        cmd = f"echo -e '{email_content}' | sendmail -t"
        logger.info(f"Executing command: {cmd}")
        
        # Execute the command
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        
        if result.stderr:
            logger.warning(f"Sendmail produced warnings: {result.stderr}")
        
        logger.info(f"Email sent from {sender} to {recipient} with subject: {subject}")
        return f"Email sent successfully from {sender} to {recipient}"
    
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
    """Check for the most recent unread email from REMOTE_USER_EMAIL in the fb_agent user's mailbox.
    This function checks a Maildir-format mailbox in /home/fb_agent/var/mail/
    
    This function assumes the main application user has read access to fb_agent's mailbox.
        
    Returns:
        Dictionary containing the most recent unread message details or empty dict if no unread messages
    """
    try:
        # Use the fb_agent user's maildir
        maildir_path = DEFAULT_MAILBOX_PATH
        remote_email = os.getenv("REMOTE_USER_EMAIL")
        
        if not remote_email:
            logger.error("REMOTE_USER_EMAIL not found in environment variables")
            return {}
        
        # Check if maildir exists
        new_mail_dir = os.path.join(maildir_path, "new")
        if not os.path.exists(new_mail_dir):
            logger.error(f"No maildir found at {new_mail_dir}")
            return {}
        
        # Check if we have read access to the maildir
        if not os.access(new_mail_dir, os.R_OK):
            error_msg = f"No read access to {new_mail_dir}. Make sure the current user has proper permissions."
            logger.error(error_msg)
            return {"error": error_msg}
        
        # List all files in the new mail directory
        mail_files = [f for f in os.listdir(new_mail_dir) if os.path.isfile(os.path.join(new_mail_dir, f))]
        
        if not mail_files:
            logger.info("No new mail files found")
            return {}
        
        # Sort mail files by modification time (newest first)
        mail_files.sort(key=lambda f: os.path.getmtime(os.path.join(new_mail_dir, f)), reverse=True)
        
        # Variables to track the most recent message
        most_recent_message = None
        most_recent_file = None
        
        # Process mail files to find the first one from the target email
        for mail_file in mail_files:
            file_path = os.path.join(new_mail_dir, mail_file)
            
            try:
                # Read the mail file
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    mail_content = f.read()
                
                # Extract headers and body
                headers, _, body = mail_content.partition('\n\n')
                headers = headers.split('\n')
                
                # Extract From header
                from_addr = ""
                subject = "No Subject"
                date_str = ""
                
                for header in headers:
                    if header.startswith('From:'):
                        from_addr = header[5:].strip()
                    elif header.startswith('Subject:'):
                        subject = header[8:].strip()
                    elif header.startswith('Date:'):
                        date_str = header[5:].strip()
                
                # Check if message is from the target email
                if remote_email.lower() not in from_addr.lower():
                    continue
                
                # Found a matching email
                most_recent_message = {
                    "from": from_addr,
                    "subject": subject,
                    "date": date_str,
                    "body": body.strip()
                }
                
                most_recent_file = file_path
                break  # Stop after finding the first matching email
                
            except Exception as e:
                logger.error(f"Error reading mail file {file_path}: {e}")
                continue
        
        # Check if we have write access to move the file to 'cur' to mark as read
        cur_mail_dir = os.path.join(maildir_path, "cur")
        can_mark_as_read = os.access(new_mail_dir, os.W_OK) and os.path.exists(cur_mail_dir) and os.access(cur_mail_dir, os.W_OK)
        
        # Mark the message as read by moving it from 'new' to 'cur'
        if most_recent_file is not None and most_recent_message is not None and can_mark_as_read:
            try:
                # Create a new filename with the :2,S suffix (S = Seen flag in Maildir)
                base_name = os.path.basename(most_recent_file)
                new_file_path = os.path.join(cur_mail_dir, base_name + ":2,S")
                
                # Move the file from 'new' to 'cur' with the seen flag
                os.rename(most_recent_file, new_file_path)
                logger.info(f"Marked email with subject '{most_recent_message['subject']}' as read")
            except Exception as e:
                logger.warning(f"Could not mark email as read: {e}. The email was still read successfully.")
        
        elif most_recent_message is not None and not can_mark_as_read:
            logger.warning(f"Email found but could not be marked as read: No write access to maildir")
        
        return most_recent_message or {}
    
    except Exception as e:
        error_msg = f"Error checking mail: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

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