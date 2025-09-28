from fastapi import UploadFile

from core import conf
from core.exceptions import BizException


def single_file_handler(file: UploadFile):
    # 判断文件类型，根据文件类型，找到对应的方法提取其中文字
    if file.filename.endswith(tuple(conf.engine.support_file_types)):
        raise BizException(f"Unsupported file type: {file.filename}")
    # 文字经过正则引擎，记录每种正则的命中个数

    # 文字经过 ac

    # 文字经过模型扫描

    # 结果记录在数据库

    # 请求规则引擎
    pass


