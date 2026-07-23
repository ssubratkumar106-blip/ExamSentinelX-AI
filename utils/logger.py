"""utils/logger.py — Structured Logging Configuration"""

import logging
import os
from datetime import datetime

def setup_logging(app):
    """Configure application-wide logging."""
    os.makedirs('logs', exist_ok=True)
    
    log_file = f'logs/examsentinelx.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    app.logger.setLevel(logging.INFO)
    return logging.getLogger(__name__)
