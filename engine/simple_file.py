import os
import shutil

from fastapi import UploadFile
from nanoid import generate

from common import conf
from common.exceptions import BizException
from common.logger import log
from core.reg import match_patterns
from core.req import rule_engine_req, llm_req
from core.reader import read
from engine import detector


def single_file_handler(file: UploadFile, task_id: str = None) -> dict:
    # 判断文件类型，根据文件类型，找到对应的方法提取其中文字
    if file.filename.endswith(tuple(conf.engine.support_file_types)):
        raise BizException(f"Unsupported file type: {file.filename}")
    filepath = os.path.join(conf.tmp_upload_dir, file.filename)
    if not task_id:
        task_id = generate()
    log.info(f"start processing task {task_id}: {file.filename} >>> {filepath}")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = read(filepath)
    # 文字经过正则引擎，记录每种正则的命中个数
    res_reg = match_patterns(text)
    # 文字经过 ac
    res_ac = detector.ac_task(task_id, text, file.filename)
    # 文字经过模型扫描
    res_bert = detector.bert_task(task_id, text, file.filename)
    # 请求规则引擎
    resp = rule_engine_req({
        "res_reg": res_reg,
        "res_ac": res_ac,
        "res_bert": res_bert,
    })
    # 请求 llm，写 prompt，询问大语言模型的改进意见等
    suggestion = llm_req(resp)
    resp["suggestion"] = suggestion
    return resp
