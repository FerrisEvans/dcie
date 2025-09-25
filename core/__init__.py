from .logger import logger as log
from .settings import Settings

conf = Settings()

__all__ = ["conf", "log"]