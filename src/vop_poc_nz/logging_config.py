import logging
import os
import sys


def setup_logging(
    output_dir: str = None, log_file: str = "analysis.log", level: int = logging.INFO
):
    """
    Configure logging for the application.

    Args:
        output_dir: Directory to save the log file. If None, uses current directory.
        log_file: Name of the log file.
        level: Logging level for the console handler. File handler is always DEBUG.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture everything at root level

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Formatter for file (detailed)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Formatter for console (clean)
    console_formatter = logging.Formatter("%(message)s")

    # File Handler
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file)
    else:
        log_path = log_file

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Log file: {os.path.abspath(log_path)}")
