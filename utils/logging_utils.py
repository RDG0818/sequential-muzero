# In utils/logging_utils.py

import sys
from loguru import logger

DEBUG_MODE = True 

logger.level("INFO", color="<white>")
logger.level("DEBUG", color="<cyan>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("CRITICAL", color="<white><bg red>")

logger.remove()

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<level>{message}</level>"
)

logger.add(
    sys.stderr,
    level="INFO",
    format=log_format,
    colorize=True
)

if __name__ == "__main__":
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")