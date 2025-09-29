from .ac import enhance_detect as ac_detect
from .bert import TextClassifier

classifier = TextClassifier()

__all__ = ["ac_detect", "classifier"]