from .asr import Asr
from .pd import Ocr

# 创建单例实例，外部只需 from engine.reader import asr
asr = Asr()
ocr = Ocr()

__all__ = ["asr", "ocr"]