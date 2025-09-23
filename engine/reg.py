import yaml
import re
import time
from pathlib import Path

CONFIG_FILE = Path("privacy.yaml")

def load_patterns():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def match_text(text, patterns):
    results = []
    for category, rules in patterns.items():
        for rule in rules:
            pattern = re.compile(rule["pattern"])
            if pattern.search(text):
                results.append({
                    "category": category,
                    "title": rule["title"],
                    "match": pattern.findall(text)
                })
    return results

if __name__ == "__main__":
    # 模拟热加载
    last_mtime = 0
    mtime = CONFIG_FILE.stat().st_mtime
    if mtime != last_mtime:  # 文件有修改
        print("配置文件更新，重新加载正则表达式...")
        patterns = load_patterns()
        last_mtime = mtime

    # 测试输入
    text = "张三的身份证号是 11010519491231002X"
    matches = match_text(text, patterns)
    if matches:
        print("检测结果:", matches)