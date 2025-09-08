import logging
import os
import sys
from pathlib import Path

from loguru import logger as log

from config.settings import conf


class InterceptHandler(logging.Handler):
    """
    日志拦截处理器：将所有 Python 标准日志重定向到 Loguru

    工作原理：
    1. 继承自 logging.Handler
    2. 重写 emit 方法处理日志记录
    3. 将标准库日志转换为 Loguru 格式
    """

    def emit(self, record: logging.LogRecord) -> None:
        # 尝试获取日志级别名称
        try:
            level = log.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 获取调用帧信息
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # 使用 Loguru 记录日志
        log.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage()
        )


async def setup_logging():
    # 移除默认处理器
    log.configure(extra={"request_id": ""})
    log.remove()
    # 日志格式
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        # "process [<cyan>{process}</cyan>] | "
        "<level>{level:^8}</level> | "
        "<cyan>{name}</cyan>:<magenta>{function}</magenta>:<blue>{line}</blue> - "
        "<level>{message}</level>"
    )
    log.add(
        sys.stdout,
        format=log_format,
        level="DEBUG" if conf.debug else "INFO",
        enqueue=True,  # 异步写入
        backtrace=True,  # 完整异常回溯
        diagnose=True,  # 变量值等诊断信息
        colorize=True,
    )

    log_dir = Path(conf.base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log.add(
        str(log_dir / "all_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level="INFO",
        rotation=conf.log.rotation,
        retention=conf.log.retention,
        compression=conf.log.compression,
        encoding="UTF-8",
        enqueue=True,
    )
    log.add(
        str(log_dir / "error_{time:YYYY-MM-DD}.log"),
        format=log_format,
        level="ERROR",
        rotation=conf.log.rotation,
        retention=conf.log.retention,
        compression=conf.log.compression,
        encoding="UTF-8",
        enqueue=True,
    )
    # 配置标准库日志
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    # 配置第三方库日志
    for logger_name in [
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.asgi",
        "fastapi",
        "fastapi.error",
    ]:
        _logger = logging.getLogger(logger_name)
        _logger.handlers = [InterceptHandler()]
        _logger.propagate = False
    # trace debug info success warning error critical
    log.success("Logger init successfully.")
