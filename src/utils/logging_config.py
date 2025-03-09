"""
Logging configuration for the NewsCrawler system.
This module provides a centralized configuration for logging across the application.
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import traceback

# Default log levels for different components
DEFAULT_LOG_LEVELS = {
    "src": logging.INFO,
    "src.api": logging.INFO,
    "src.scrapers": logging.INFO,
    "src.database": logging.INFO,
    "src.vector": logging.INFO,
    "src.llm": logging.INFO,
    "src.utils": logging.INFO,
    "src.models": logging.INFO,
}

# Log format strings
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
JSON_FORMAT = {
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s",
    "logger": "%(name)s",
    "file": "%(filename)s",
    "line": "%(lineno)d",
    "message": "%(message)s"
}


class JsonFormatter(logging.Formatter):
    """
    Formatter for JSON-formatted logs.
    """
    
    def __init__(self, fmt_dict: Dict[str, str] = None):
        """
        Initialize the JSON formatter.
        
        Args:
            fmt_dict: Dictionary mapping log record attributes to output fields
        """
        super().__init__()
        self.fmt_dict = fmt_dict or JSON_FORMAT
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        record_dict = {}
        
        # Apply standard formatting to all fields
        for key, fmt in self.fmt_dict.items():
            record_dict[key] = logging.Formatter(fmt).format(record)
        
        # Add exception info if present
        if record.exc_info:
            record_dict["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, "extra"):
            record_dict["extra"] = record.extra
        
        return json.dumps(record_dict)


class ContextAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to log messages.
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message to add context.
        
        Args:
            msg: Log message
            kwargs: Keyword arguments for the logger
            
        Returns:
            Tuple of (modified message, modified kwargs)
        """
        # Add context to the message
        context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
        if context_str:
            msg = f"{msg} [{context_str}]"
        
        return msg, kwargs


def configure_logging(
    log_level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_to_json: bool = False,
    json_log_file: Optional[str] = None,
    component_log_levels: Optional[Dict[str, Union[int, str]]] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Default log level
        log_file: Path to log file (None for no file logging)
        log_to_console: Whether to log to console
        log_to_json: Whether to log in JSON format
        json_log_file: Path to JSON log file (None for no JSON logging)
        component_log_levels: Dictionary mapping component names to log levels
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure console logging
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(CONSOLE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Configure file logging
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(FILE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure JSON logging
    if log_to_json and json_log_file:
        # Create directory if it doesn't exist
        json_log_dir = os.path.dirname(json_log_file)
        if json_log_dir and not os.path.exists(json_log_dir):
            os.makedirs(json_log_dir)
        
        # Create rotating file handler
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        json_handler.setLevel(log_level)
        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_handler)
    
    # Configure component-specific log levels
    component_levels = {**DEFAULT_LOG_LEVELS, **(component_log_levels or {})}
    for component, level in component_levels.items():
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        component_logger = logging.getLogger(component)
        component_logger.setLevel(level)
    
    # Log configuration
    logging.info(f"Logging configured with level {logging.getLevelName(log_level)}")
    if log_file:
        logging.info(f"Logging to file: {log_file}")
    if log_to_json and json_log_file:
        logging.info(f"Logging to JSON file: {json_log_file}")


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get a logger with context.
    
    Args:
        name: Logger name
        **context: Context key-value pairs
        
    Returns:
        Logger with context
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextAdapter(logger, context)
    
    return logger


def log_exception(logger: logging.Logger, e: Exception, message: str = "An error occurred") -> None:
    """
    Log an exception with traceback.
    
    Args:
        logger: Logger to use
        e: Exception to log
        message: Message to log
    """
    logger.error(f"{message}: {str(e)}", exc_info=True)


class RequestIdFilter(logging.Filter):
    """
    Filter that adds request ID to log records.
    """
    
    def __init__(self, request_id: str):
        """
        Initialize the filter.
        
        Args:
            request_id: Request ID to add
        """
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records and add request ID.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record
        """
        record.request_id = self.request_id
        return True 