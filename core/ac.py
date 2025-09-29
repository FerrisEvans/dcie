import os
from typing import Any

from ahocorasick import Automaton

from common import conf
from common.logger import log

# 分类敏感词库（示例）
sensitive_words = {}
for filename in os.listdir(conf.sensitive_words_dict_path):
    if filename.endswith(".txt"):
        category = os.path.splitext(filename)[0]  # 去掉后缀，比如 "涉政"
        filepath = os.path.join(conf.sensitive_words_dict_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
            sensitive_words[category] = words

# 构建AC自动机快速匹配
automaton = Automaton()
for category, words in sensitive_words.items():
    for word in words:
        automaton.add_word(word, (category, word))
automaton.make_automaton()

def _scan_text(text):
    """文本内容扫描"""
    results = []
    if not text:
        return results
    try:
        # AC自动机扫描
        for end_idx, (category, word) in automaton.iter(text):
            start_idx = end_idx - len(word) + 1
            results.append({
                "text": text[start_idx:end_idx + 1],
                "category": category,
                "position": (start_idx, end_idx),
            })
    except Exception as e:
        log.error(f"AC automation failed: [{repr(e)}]")
        log.exception(e)

    return results


def enhance_detect(text) -> list[dict[str, Any]]:
    results = []
    if not text:
        return results
    # 将文本按换行符分割成多行，并记录行号
    lines = text.split('\n')
    # 行号从1开始
    for line_num, line in enumerate(lines, 1):
        line = line.strip()  # 去除首尾空白
        if not line:
            continue  # 跳过空行

        # --- 每行的基础敏感词扫描 ---
        line_hits = _scan_text(line)
        # 添加行号信息到命中结果
        for hit in line_hits:
            hit["line"] = line_num
        results.extend(line_hits)
    return results
