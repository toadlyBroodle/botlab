# Installing BotLab Dependencies

This directory contains a script to easily set up a virtual environment and install all Python dependencies required by BotLab.

## Quick Installation

Run the following commands to set up the environment and install all dependencies:

```bash
# Make the script executable (if needed)
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

## What Does It Do?

The script:
1. Creates a `.venv` directory at the root of the project
2. Installs all dependencies from `requirements.txt`
3. Installs local packages (`agents` and `swarms`) in development mode

## Usage

After installation, activate the virtual environment:

```bash
source .venv/bin/activate
```

To deactivate the virtual environment when you're done:

```bash
deactivate
```

## Troubleshooting

If you encounter permission issues, make sure the script is executable:

```bash
chmod +x setup_env.sh
```

For manual installation:

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install local packages
pip install -e ./agents
#pip install -e ./swarms
``` 

In case of dependency issues, you can use Poetry to install dependencies with the `pyproject.toml` file or `install_dependencies.sh` script, 
which will also export the dependencies to `requirements.txt` if you still want to use pip.

## Setting Up User Feedback Agent

The user feedback agent requires a dedicated system user (`fb_agent`) for handling email communication. Follow these steps to set it up:

### 1. Create the fb_agent User

```bash
# Create the new user without login capabilities
sudo adduser --disabled-password --gecos "Feedback Agent" fb_agent

# Add user to mail group
sudo usermod -a -G mail fb_agent
```

### 2. Set Up Email Configuration

Configure email forwarding for the fb_agent user:

```bash
# Set up email forwarding in Postfix virtual file
sudo sh -c 'echo "fb_agent@botlab.dev fb_agent" >> /etc/postfix/virtual'
sudo postmap /etc/postfix/virtual

# Create proper Maildir structure for fb_agent
sudo mkdir -p /home/fb_agent/var/mail/{new,cur,tmp}
sudo chown -R fb_agent:mail /home/fb_agent/var/mail

# Set permissions for the Maildir and its contents
# fb_agent user has full control, 'mail' group has read/write/execute
sudo find /home/fb_agent/var/mail -type d -exec chmod 770 {} \;
sudo find /home/fb_agent/var/mail -type f -exec chmod 660 {} \;

# Set the setgid bit on Maildir directories to ensure new files inherit the 'mail' group
sudo chmod g+s /home/fb_agent/var/mail
sudo chmod g+s /home/fb_agent/var/mail/new
sudo chmod g+s /home/fb_agent/var/mail/cur
sudo chmod g+s /home/fb_agent/var/mail/tmp

# Update Postfix configuration if needed
sudo postconf -e "home_mailbox = var/mail/"
sudo systemctl restart postfix
```

### 3. Set Up Permissions for Your Application User

Add your application user to the mail group so it can read the fb_agent's mailbox:

```bash
# Replace 'your_username' with your actual username
sudo usermod -a -G mail your_username

# You'll need to log out and back in for this change to take effect

# After logging back in, check if you can access the maildir
ls -la /home/fb_agent/var/mail/
```

### 4. Configure Environment Variables

Add the following to your `.env` file:

```
# LOCAL_USER_EMAIL is used as the sender address in outgoing emails
LOCAL_USER_EMAIL=fb_agent@botlab.dev

# REMOTE_USER_EMAIL is the external user's email address for sending reports and receiving commands
REMOTE_USER_EMAIL=your_actual_email@example.com
```

IMPORTANT: Make sure to set REMOTE_USER_EMAIL to your actual external email address where you want to receive agent updates and from which you'll send commands.

### 5. Test the Setup

Run the example script to test if everything is properly configured:

```bash
python -m agents.user_feedback.example
```

The output should show that:
- The fb_agent user exists
- The maildir exists and is accessible
- Your user has at least read access to the mailbox
- Your REMOTE_USER_EMAIL is properly configured

To test email sending directly:

```bash
# Send a test email using the mail command
echo "Test email body" | mail -s "Test Email Subject" -r "fb_agent@botlab.dev" your_actual_email@example.com

# Check the mail logs to see if it was sent
tail /var/log/mail.log
```

### 6. Email Workflow

With this setup:
1. You can send an email to fb_agent@botlab.dev with commands in the body
2. The UserFeedbackAgent will check for emails from your REMOTE_USER_EMAIL address
3. Progress reports will be sent to your REMOTE_USER_EMAIL address

Example commands in emails:
```
FREQUENCY: 2
FEEDBACK: Looking good, but please include more details on X
FOCUS: research
```