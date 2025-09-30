from pathlib import Path
import requests
from requests import Response
from requests.auth import HTTPBasicAuth

from common import conf
from common.logger import log

def rule_engine_req(data=None) -> Response | None:
    if not data:
        log.warning("Failed to send request to rule engine. Parameter is null.")
        return None
    try:
        log.info(f"Rule engine req: {conf.engine.rule_engine_api}, params: {data}")
        resp = requests.post(conf.engine.rule_engine_api, json=data, auth=HTTPBasicAuth("bba", "123"))
        log.success(f"HTTP response: {resp.json()}")
        return resp
    except Exception as e:
        log.error(f"rule_engine_api error: {e}")
        return None


def beacon_req(data) -> Response:
    if not data:
        log.warning("Failed to send request to Beacon. Parameter is null.")

    # 获取当前脚本所在目录
    base_dir = Path(__file__).parent
    # 拼接出 access_code 文件路径
    file_path = base_dir / "access_code"
    # 读取文件内容
    with file_path.open("r", encoding="utf-8") as f:
        access_code = f.read().strip()  # 去掉前后空格和换行


    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_code}",
    }
    body = {
        "model": "DeepSeek-V3-0324",
        "message": [
            {
                "role": "system", # system assistant user
                "content": "you are helpful agent",
            },
            {
                "role": "user",
                "content": "hello, what's your name",
            }
        ],
        "stream": False,
    }
    response = requests.post(conf.engine.beacon_llm_api, json=body, headers=headers)
    log.success(f"HTTP response: {resp.json()}")
    return response


if __name__ == "__main__":
    resp = rule_engine_req({
        "task_id": "101010",
        "sensitiveWordRes": {},
        "bertRes": [{}]
    })