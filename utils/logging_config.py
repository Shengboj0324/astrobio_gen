#!/usr/bin/env python3
"""
Unicode-Safe Logging Configuration for Windows
=============================================

Fixes UnicodeEncodeError issues with emoji characters in logging on Windows systems.
Provides fallback mechanisms and proper encoding configuration.
"""

import logging
import os
import sys
from typing import Optional


class UnicodeAwareFormatter(logging.Formatter):
    """Logging formatter that handles Unicode characters safely on Windows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emoji_fallbacks = {
            "[START]": "[START]",
            "[OK]": "[OK]",
            "[FAIL]": "[FAIL]",
            "[WARN]": "[WARN]",
            "[PROC]": "[PROC]",
            "[DATA]": "[DATA]",
            "[AI]": "[AI]",
            "[PKG]": "[PKG]",
            "[NET]": "[NET]",
            "[SEARCH]": "[SEARCH]",
            "[CHART]": "[METRIC]",
            "[TARGET]": "[TARGET]",
            "[BIO]": "[BIO]",
            "[MICRO]": "[MICRO]",
            "[GALAXY]": "[SPACE]",
            "[SUCCESS]": "[SUCCESS]",
            "[SAVE]": "[SAVE]",
            "[COMPLETE]": "[COMPLETE]",
            "[EARTH]": "[PLANET]",
            "💬": "[CHAT]",
            "[NOTE]": "[NOTE]",
            "[FIX]": "[CONFIG]",
            "📁": "[FILE]",
            "[SHINE]": "[STAR]",
            "⬆️": "[UP]",
            "[BOARD]": "[LIST]",
            "🎪": "[DEMO]",
            "[LINK]": "[LINK]",
            "[MASK]": "[MASK]",
            "🎨": "[ART]",
            "🎵": "[SOUND]",
            "🎬": "[VIDEO]",
            "🎮": "[GAME]",
            "🎲": "[RANDOM]",
            "🎰": "[SLOT]",
        }

    def format(self, record):
        """Format log record with Unicode safety."""
        try:
            # First try normal formatting
            formatted = super().format(record)

            # Test if the formatted message can be encoded with the console encoding
            if sys.platform.startswith("win"):
                try:
                    formatted.encode("cp1252", errors="strict")
                    return formatted
                except UnicodeEncodeError:
                    # Replace emoji characters with text fallbacks
                    for emoji, fallback in self.emoji_fallbacks.items():
                        formatted = formatted.replace(emoji, fallback)
                    return formatted
            else:
                return formatted

        except Exception as e:
            # Fallback to safe ASCII representation
            safe_msg = str(record.getMessage()).encode("ascii", errors="replace").decode("ascii")
            return f"{record.levelname}: {safe_msg}"


def setup_unicode_safe_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Setup Unicode-safe logging configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to log to (UTF-8 encoded)
    """

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = UnicodeAwareFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler with Unicode safety
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # On Windows, try to set UTF-8 mode if possible
    if sys.platform.startswith("win"):
        try:
            # Try to enable UTF-8 mode
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            elif hasattr(sys.stdout, "encoding"):
                # If we can't reconfigure, we'll rely on the formatter fallbacks
                pass
        except Exception:
            # If all else fails, the formatter will handle the conversion
            pass

    root_logger.addHandler(console_handler)

    # File handler with UTF-8 encoding if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")

    root_logger.setLevel(level)


def get_unicode_safe_logger(name: str) -> logging.Logger:
    """Get a logger that's been configured for Unicode safety."""
    return logging.getLogger(name)


# Auto-configure logging when module is imported
if __name__ != "__main__":
    setup_unicode_safe_logging()

    # Also configure all existing loggers
    import logging

    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(
                        UnicodeAwareFormatter(
                            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        )
                    )
