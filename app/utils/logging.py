"""
Logging configuration for the Agentic File Explorer.

Usage:
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Starting query", extra={"task": task})
"""

import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with consistent formatting.

    All loggers share the same handler and format so output
    is uniform across all modules.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


# Root application logger
log = get_logger("app")
