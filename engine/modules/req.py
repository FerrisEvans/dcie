# from conf import api_active, rule_engine_uri
# from log import logging
#
# import requests
# from requests.auth import HTTPBasicAuth
#
# def rule_engine_api(data={}):
#     try:
#         print("rule_engine_api")
#         if not api_active:
#             return data
#         print(f"sending HTTP request: {rule_engine_uri}, params: {data}")
#         resp = requests.post(rule_engine_uri, json=data, auth=HTTPBasicAuth("bba", "123"))
#         print(f"HTTP response: {resp.json()}")
#         return resp
#     except Exception as e:
#         logging.error(f"rule_engine_api error: {e}")
#         return data
#
#
# if __name__ == "__main__":
#     resp = rule_engine_api({
#         "task_id": "101010",
#         "sensitiveWordRes": {},
#         "bertRes": [{}]
#     })