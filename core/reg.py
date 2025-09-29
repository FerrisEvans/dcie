import yaml
import re
from collections import defaultdict
from typing import Any, Dict

from common import conf
from common.logger import log

def _load_patterns():
    all_rules = []
    for file in conf.reg_path.glob("*.yaml"):
        with file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                all_rules.extend(data)
    return all_rules


def match_patterns(text: str) -> Dict[str, Any]:
    """
    输入文本，返回匹配到的规则详情以及每个规则(title)的命中次数
    todo 正则匹配并不精准
    :param text: 待匹配文本
    :return: {
        "matches": [
            {"category": ..., "title": ..., "pattern": ..., "match": "实际匹配到的内容"},
            ...
        ],
        "title_count": {
            "规则名1": 匹配次数,
            "规则名2": 匹配次数,
        }
    }
    """
    matches = []
    title_count = defaultdict(int)
    rules = _load_patterns()
    for category_item in rules:
        category = category_item.get("category")
        for rule in category_item.get("rules", []):
            title = rule.get("title")
            pattern = rule.get("pattern")

            if not pattern:  # 忽略空规则
                continue

            try:
                regex = re.compile(pattern)
            except re.error as e:
                log.warning(f"{pattern} is illegal: {e}")
                continue

            for m in regex.finditer(text):
                matches.append({
                    "category": category,
                    "title": f"{category}:{title}",
                    "pattern": pattern,
                    "match": m.group(0)
                })
                title_count[title] += 1

    return {
        "matches": matches,
        "title_count": dict(title_count)
    }
