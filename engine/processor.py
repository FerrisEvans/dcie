# from conf import context_chars, threshold, support_suffix
# from log import logging
#
# import classifier
# import db
# import json
# import reader
# import scan
#
#
# def support(filename):
#     flag = False
#     for s in support_suffix:
#         if filename.endswith(f".{s}"):
#             flag = True
#             break
#     return flag
#
# # Global initialization 避免在循环中加载 classifier
# # cl = classifier.TextClassifier()
#
# def sensitive_word(task_id, file_path, filename) -> tuple[db.SensitiveWord, bool]:
#     """
#     Args:
#         task_id: e.g. 1117264
#         context_path: e.g. /mnt/scan/ai
#         filename: e.g. F78CS0124.push
#     Returns:
#         1. An instance of Model[db.SensitiveWord]
#         2. If this record is toxic
#     """
#     if not support(filename):
#         return None, False
#
#     res_ac, res = scan.scan_file(file_path)
#     logging.debug(f">>> scanning sensitive words >>> {file_path} \n res: {res} \n res_ac: {res_ac}")
#     toxic = True if res or res_ac else False
#     s = db.SensitiveWord(task_id=task_id, filename=filename, res=json.dumps(res, ensure_ascii=False), res_ac=json.dumps(res_ac, ensure_ascii=False), toxic=toxic)
#     return s, toxic
#
# def bert(task_id, file_path, filename) -> tuple[list[db.Bert], bool]:
#     """
#     Args:
#         task_id: e.g. 1117264
#         context_path: e.g. /mnt/scan/ai
#         filename:  e.g. F78CS0124.push
#     Returns:
#         1. A instances list of Model[db.Bert]
#         2. If this record is toxic
#     """
#
#     print("------------------------bert----------------------")
#     arr = []
#     toxic = False
#
#     if not support(filename):
#        return arr, toxic
#
#     text = reader.extract_text(file_path)
#     print(f"text: {text}")
#     if not text:
#         return arr, toxic
#
#     # predict & trace
#     cl = classifier.TextClassifier()
#     # result = cl.predict(text)
#     # print(f"result: {result}")
#     # print(cl.trace(original_text=text, chunks=result['chunks'], chunk_probs=result['chunk_probs']))
#     # for problem in cl.trace(original_text=text, chunks=result['chunks'], chunk_probs=result['chunk_probs']):
#     #     score = problem['score']
#     #     toxic = True if score and score >= threshold else False
#     #     b = db.Bert(task_id=task_id, filename=filename, label=problem['label'], score=score, start_pos=problem['start_pos'], end_pos=problem['end_pos'], context=problem['context'][:context_chars], toxic=toxic)
#     #     print(f">>> bert predict >>> {file_path} \n filename: {filename} \n label: {problem['label']} >>> score: {score}")
#     #     arr.append(b)
#     problems = cl.trace(text=text)
#     for problem in problems:
#         score = problem['score']
#         toxic = True if score and score >= threshold else False
#         b = db.Bert(task_id=task_id, filename=filename, label=problem['label'], score=score, start_pos=problem['start_pos'], end_pos=problem['end_pos'], context=problem['context'][:context_chars], toxic=toxic)
#         print(f">>> bert predict >>> {file_path} \n filename: {filename} \n label: {problem['label']} >>> score: {score}")
#         arr.append(b)
#     return arr, toxic
