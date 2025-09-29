from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from common.logger import log

# 初始化全局ASR处理器（单例模式）
class Asr:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        log.info(f"Lazy init model: %s. Load only on first use.")
        try:
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={ "max_single_segment_time": 30000 },
                device="cpu",
                disable_update=True
            )
        except Exception as e:
            log.error(f"ASR模型初始化失败: {str(e)}")
            self.model = None

    def transcribe(self, audio_path):
        """安全执行语音转录"""
        if not self.model:
            raise RuntimeError("ASR模型未就绪")

        try:
            res = self.model.generate(
                input=audio_path,
                language="auto",
                use_itn=True,
                batch_size_s=60
            )
            return rich_transcription_postprocess(res[0]["text"])
        except Exception as e:
            log.error(f"音频处理错误: {str(e)}")
            return ""

