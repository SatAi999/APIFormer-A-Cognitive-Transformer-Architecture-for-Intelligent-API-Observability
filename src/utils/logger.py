import logging
from rich.logging import RichHandler

def setup_logger(name: str = "APIFormer+", level: int = logging.INFO) -> logging.Logger:
    """Sets up a rich colorized logger for the APIFormer+ framework."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

# Shared logger instance
logger = setup_logger()
