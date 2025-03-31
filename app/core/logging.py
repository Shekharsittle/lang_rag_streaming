"""
logging.py - Logging configuration for the application
"""

import logging
import sys
from pathlib import Path
from app.core.config import settings

def configure_logging() -> None:
    """
    Configure logging for the application.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper())
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure log file path
    log_file = log_dir / "app.log"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler
            logging.FileHandler(
                filename=log_file,
                mode='a',  # append mode
                encoding='utf-8'
            )
        ],
    )
    
    # Set level for external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    # Log startup message
    logging.info(f"Logging initialized. Log file: {log_file}")