import os

from ahocorasick import Automaton

from core import conf

# 分类敏感词库（示例）
sensitive_words = {}
for filename in os.listdir(conf.sensitive_words_dict_path):
    if filename.endswith(".txt"):
        category = os.path.splitext(filename)[0]  # 去掉后缀，比如 "涉政"
        filepath = os.path.join(conf.sensitive_words_dict_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
            sensitive_words[category] = words

print(sensitive_words)

# 构建AC自动机快速匹配
automaton = Automaton()
for category, words in sensitive_words.items():
    for word in words:
        automaton.add_word(word, (category, word))
automaton.make_automaton()


def scan_text(text, file_path=None):
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
                "file": file_path or ""
            })
    except Exception as e:
        print(f"扫描异常: {str(e)}")

    return results


def enhance_detect(text, file_path):
    results = []
    if not text:
        return results
    # 将文本按换行符分割成多行，并记录行号
    lines = text.split('\n')

    for line_num, line in enumerate(lines, 1):  # 行号从1开始
        line = line.strip()  # 去除首尾空白
        if not line:
            continue  # 跳过空行

        # --- 每行的基础敏感词扫描 ---
        line_hits = scan_text(line, file_path)  # 假设scan_text支持单行扫描
        # 添加行号信息到命中结果
        for hit in line_hits:
            hit["line"] = line_num  # 记录敏感词所在行号
        results.extend(line_hits)

        # --- 每行的模型风险判断 ---
        # 清理换行符（防止行内有残留的换行符）
        # cleaned_line = line.replace('\n', ' ')
        # 模型预测
        # labels, probs = model.predict(cleaned_line)
        # print(labels, probs,cleaned_line)

        # if "normal" not in labels[0] and probs[0] > 0.9:
        #     # print(cleaned_line)
        #     results.append({
        #         "type": "context_risk",
        #         "score": probs[0],
        #         "line": line_num,  # 记录风险行号
        #         "content": line  # 可选：记录风险文本片段
        #     })

    return results


# def scan_file(file_path):
#     results = []
#     text = reader.extract_text(file_path)  # 调用自定义的文本提取方法
#     logging.debug(f"*********** >> text << *********** \n {text} \n **********************************")
#     resultsAC, results = enhance_detect(text, file_path)
#     return resultsAC, results
