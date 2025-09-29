import requests
from requests.auth import HTTPBasicAuth

from common import conf
from common.logger import log

def rule_engine_req(data=None):
    if not data:
        log.warning("Failed to send request to rule engine. Parameter is null.")
        return {}
    try:
        log.info(f"sending HTTP request: {conf.engine.rule_engine_api}, params: {data}")
        resp = requests.post(conf.engine.rule_engine_api, json=data, auth=HTTPBasicAuth("bba", "123"))
        log.success(f"HTTP response: {resp.json()}")
        return resp
    except Exception as e:
        log.error(f"rule_engine_api error: {e}")
        return {}


def llm_req(data) -> str:
    if not data:
        log.warning("Failed to send request to llm. Parameter is null.")

    return ""


if __name__ == "__main__":
    resp = rule_engine_req({
        "task_id": "101010",
        "sensitiveWordRes": {},
        "bertRes": [{}]
    })