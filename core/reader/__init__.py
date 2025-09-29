from .asr import Asr
from .pd import Ocr
from .loader import extract_text as read

# 创建单例实例，外部只需 from engine.reader import asr
asr = Asr()
ocr = Ocr()

__all__ = ["asr", "ocr", "read"]