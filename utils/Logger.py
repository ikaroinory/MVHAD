import sys
from pathlib import Path

from loguru import logger


class Logger:
    @staticmethod
    def init(log_name: str | Path = None):
        format_str = '<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> <level>{message}</level>'
        logger.remove()
        logger.add(sys.stdout, format=format_str)

        if log_name is not None:
            logger.add(log_name, format=format_str)

    @staticmethod
    def info(message: str):
        logger.info(message)

    @staticmethod
    def debug(message: str):
        logger.debug(message)

    @staticmethod
    def warning(message: str):
        logger.warning(message)

    @staticmethod
    def error(message: str):
        logger.error(message)

    @staticmethod
    def critical(message: str):
        logger.critical(message)
