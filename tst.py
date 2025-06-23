import logging

# ANSI escape codes for colors
COLOR_DEBUG = '\033[94m'  # Blue
COLOR_INFO = '\033[92m'   # Green
COLOR_WARNING = '\033[93m' # Yellow
COLOR_ERROR = '\033[91m'  # Red
COLOR_CRITICAL = '\033[41;97m' # White on Red background
COLOR_RESET = '\033[0m'   # Reset to default

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_message = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{COLOR_DEBUG}{log_message}{COLOR_RESET}"
        elif record.levelno == logging.INFO:
            return f"{COLOR_INFO}{log_message}{COLOR_RESET}"
        elif record.levelno == logging.WARNING:
            return f"{COLOR_WARNING}{log_message}{COLOR_RESET}"
        elif record.levelno == logging.ERROR:
            return f"{COLOR_ERROR}{log_message}{COLOR_RESET}"
        elif record.levelno == logging.CRITICAL:
            return f"{COLOR_CRITICAL}{log_message}{COLOR_RESET}"
        return log_message

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(colored_formatter)
logger.addHandler(handler)

logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning message.")
logger.error("This is an error message.")
logger.critical("This is a critical message.")