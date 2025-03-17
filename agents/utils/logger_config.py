import os
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging"""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, 'props'):
            log_obj.update(record.props)
            
        if record.exc_info:
            log_obj['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logger(name, log_dir='logs', disable_console=False, console_log_level=logging.DEBUG, file_log_level=logging.DEBUG):
    """Set up a logger with JSON formatting and size-based rotation
    
    Args:
        name: Name of the logger
        log_dir: Directory to store log files
        disable_console: If True, no console handler will be added
        console_log_level: Log level for console output (default: DEBUG)
        file_log_level: Log level for file output (default: DEBUG)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create logger
    logger = logging.getLogger(name)
    
    # Set the logger level to the minimum of console and file log levels
    # This ensures we only capture messages that at least one handler will process
    min_level = min(console_log_level, file_log_level)
    logger.setLevel(min_level)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create handlers
    # File handler with size-based rotation (10MB per file, keep 1 backup files)
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=1,
        encoding='utf-8'
    )
    file_handler.setLevel(file_log_level)
    
    # Create formatters and add it to handlers
    json_formatter = JsonFormatter()
    file_handler.setFormatter(json_formatter)
    
    # Add file handler
    logger.addHandler(file_handler)
    
    # Only add console handler if not disabled
    if not disable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to avoid double logging
    logger.propagate = False
    
    return logger 