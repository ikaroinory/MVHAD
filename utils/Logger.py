import sys
from pathlib import Path

from loguru import logger


class Logger:
    @staticmethod
    def init(log_name: str | Path = None):
        logger.remove()
        logger.add(sys.stdout, format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}')

        if log_name is not None:
            logger.add(
                log_name,
                format='<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> <r>|</r> <level>{level: <8}</level> <r>|</r> {message}'
            )

    @staticmethod
    def critical(message: str):
        logger.critical(message)

    @staticmethod
    def debug(message: str):
        logger.debug(message)

    @staticmethod
    def error(message: str):
        logger.error(message)

    @staticmethod
    def info(message: str):
        logger.info(message)
