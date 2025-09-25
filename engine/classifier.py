# from conf import path_conf, threshold
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
# )
#
# import torch  # For tensor operations and GPU/CPU management
# import torch.nn.functional as F  # For softmax function
#
#
# class TextClassifier:
#     def __init__(self, model_path=path_conf["model"]):
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
#         self.max_seq_length = self.model.config.max_position_embeddings  # 如512
#         self.label = list(self.model.config.id2label.values())
#
#     def _preprocess_text(self, text: str, window_size: int = 80, stride: int = 50):
#         """将文本分割为固定窗口大小的片段"""
#         tokens = self.tokenizer.tokenize(text)
#         total_length = len(tokens)
#         segments = []
#
#         # 滑动窗口分割
#         for start in range(0, total_length, stride):
#             end = start + window_size
#             if end > total_length:
#                 end = total_length
#             segment = tokens[start:end]
#             segments.append({
#                 'text': segment,
#                 'start_char': start,
#                 'end_char': end
#             })
#
#             # 如果剩余文本不足窗口大小，提前结束
#             if end == total_length:
#                 break
#
#         # 将片段转换回字符串并进行分词处理
#         tokenized_segments = [
#             {
#                 'token': self.tokenizer(
#                     self.tokenizer.convert_tokens_to_string(segment['text']),
#                     truncation=True,
#                     padding="max_length",
#                     max_length=window_size,
#                     return_tensors="pt"
#                 ),
#                 'start_char': segment['start_char'],
#                 'end_char': segment['end_char'],
#                 'text': segment['text'],
#             }
#             for segment in segments
#         ]
#         return tokenized_segments
#
#     def trace(self, text: str, window_size: int = 50, stride: int = 30, threshold=threshold):
#         """对输入文本进行分类"""
#         # 预处理文本
#         tokenized_segments = self._preprocess_text(text, window_size, stride)
#
#         problem_zones = []
#         with torch.no_grad():
#             for segment in tokenized_segments:
#                 inputs = {key: val.to(self.model.device) for key, val in segment['token'].items()}
#                 outputs = self.model(**inputs)
#                 logits = outputs.logits
#                 probs = F.softmax(logits, dim=1)  # 将 logits 转换为概率分布
#                 predicted_class = torch.argmax(probs, dim=1).item()
#                 score = float(probs[0].cpu().numpy()[predicted_class])
#                 if (score >= threshold):
#                     problem_zones.append({
#                         'start_pos': segment['start_char'],
#                         'end_pos': segment['end_char'],
#                         'label': self.label[predicted_class],
#                         'score': score,
#                         'context': self.tokenizer.convert_tokens_to_string(segment['text']),
#                     })
#         return problem_zones
#
#
# # 使用示例
# if __name__ == "__main__":
#     classifier = TextClassifier()
#     long_text = """
#
# 我的一项“长时间、高效学习”项目结果是，一套“鸢尾花书”。
#
# 下面，结合鸢尾花书的创作，和大家聊聊我长时间、高效学习的“军规”！
# 注意，以下都是个人学习经验，没有任何科学数据，没有任何大道理，请甄别。
# 长时间学习，在我看来，是以年为尺度的学习项目。
# 此外，这个回答关注长时间学习的“心理建设”，不是介绍如何读一本书、学一门课。
# 设定明确、量化、可实现的目标！
# 举个例子，“我想搞机器学习、深度学习！”这个目标显然极其空洞、宽泛！
# 那么明确、可实现的目标长成什么样子？
# 大家可能已经发现，量化非常重要。比如，学习英语，背诵多少词汇、句子、短文，英语考试具体分数。比如，备考，考到怎样的分数，考取什么样类型的学校、专业。比如，搞科研发表文章，发表几篇文章、什么影响因子的刊物。
# 做梦的时候，不妨胆子大点！
# 制定学习目标的时候，不妨大胆一些！求乎上者居乎中，求乎中者居乎下。
# 没事儿给自己画个大饼，猛灌几口鸡汤，还是有用的！
# 最开始，根本没有什么“鸢尾花书”之类的想法。当时，就是想把自己的学习笔记系统整理一下，和大家分享，让大家少走弯路。
# 没成想，清华大学出版社竟然看中 ...
# 别把自己逼得太狠，偶尔放个假！
# 以年为单位的长时间学习，就像是在山洞中抹黑前行，始终不见洞口的光亮。
# 路途中难免不安、沮丧、焦虑 ...
# 鸢尾花书的创作过程，我自己大哭过十次以上（别笑话我）。很多时候就是因为恐惧自己挖的坑之大。很多时候仅仅苦于没有创意灵感，内容枯燥乏味。
# 这时候，给自己放个假。没有必要规划旅行之类的。爬爬山、跑跑步，这些不花钱的活动都很有效果。
#
# 一些意外的惊喜可能会让你发现有些烦恼不值一提
# 长时间学习是马拉松，不是短跑。拼的是耐力、意志品质。
# 千万不要把学习目标告诉全世界！
# 想做某事 不等于 做成某事。
# 让一棵树从小苗长成大树的最好的办法，就是“偷偷摸摸”地给它施肥、除草、浇水。
# 过度的“曝光”只能让它早死。
# 大家都很“忙”，“广播”梦想只可能成为别人一时的谈资、笑料。
# 更重要的是，这个世界很残忍。你的梦想很可能被其他人（包括亲友）以“关心你、为你好”的借口，放在地上揉搓取乐。
# 让别人不再笑话你的梦想的最好办法，就是静悄悄地把梦想变成现实，然后再云淡风轻地告诉大家！
# 兴趣，是廉价的！
# 重复一遍，兴趣是廉价的！今天对弹钢琴感兴趣，明天对滑雪感兴趣，后天对哲学感兴趣，下星期想养狗，下个月想学油画 ...
# 这些“兴趣”可能转瞬即逝。
# 或者，怀着一腔热情的你进入某个领域一段时间之后，会对它索然无味。
# 长时间学习是“投资”，要考虑成本、收益，不能仅凭一时兴起，要做足功课。这一点，我们后面也要谈到。
# 分解目标，规划学习路线，设定时间节点！
# 时间，时间，时间！必须要给自己长时间学习设定节点。
# 什么时间该完成什么，把它定下来！
# 学习目标需要调整变化！
# 调整变化，不是“始乱终弃”！调整变化，是审时度势。
# 学习的过程，你的胃口可能变大。学着学着，你发现新的领域。学习时，必须制定计划。但是，学习的过程中，计划是用来修改的。
# 一边执行，一边修改计划，很正常。
# 还有一个小秘诀，把目标一笔一划写出来（换成隐喻的方式），放在一个只有自己能看得见的地方！而且要经常翻翻看看、修修改改。
# 碎片化时间、系统学习！
# 碎片、系统这两个词是矛盾的！
# 是的，但是放在长时间学习上，它们又是“绝配”。
# 大块的时间永远都是难找的！而生活中零碎的时间，无处不在！笔记本、平板、书、纸笔，这些媒介可以方便我们利用碎片化的时间。
# 我们常说，现在没有状态，不想学习。这是极其错误的心理暗示！
# 好的状态是“学出来的”，不是“等出来的”。
# 学习不是正襟危坐、沏一杯茶、焚香沐浴。
# 学习是上班赶路、狼吞虎咽、争分夺秒！
# 学习资源集中存放！
# 把纸质参考资料集中存放。电子资料，集中存放，并且至少备份两份。
# “鸢尾花书”项目大概有100G的PDF电子书，备份在好几个硬盘中，不定期更新。为了省去麻烦，去哪下载PDF电子书，我就不说了 ...
# 我的长时间学习的一个常用的手段就是，学习笔记系统电子化。笔记写成word文档，做成latex文件，Jupyter Notebook都可以！
#
# 纸质笔记，PDF草稿算是电子笔记
# 特别是，日后有发表需要的时候，你就会发现“电子化笔记”的力量！
#
# 这是一小部分有关可视化的图书
#
#
# 其中一个书架（稍微上镜的那个）
# 不怕慢，就怕站！
# 迅速开始，缓慢结束。
# 这句话，我很受用。这是我在一本书上看到的一句话，现在记不得哪本书了。
# 学起来！干起来！书要一页页读，笔记要一笔笔写。对自己要有耐心，对更宏大的目标要有耐心。
# 屏蔽诱惑，忍受孤独！
# 朋友这周要爬山，你要去图书馆。
# 亲戚十一要看海，你要去图书馆。
# 同事下午要逛街，你要去图书馆。
# 室友晚上要烧烤，你要去图书馆。
# 你不是一个不食人间烟火的怪物，你的心里装着一个小小的、不为人知的梦想。
# 为了实现梦想，你不得不取舍、牺牲。
#
# 很喜欢的图书馆的一个窗口。一年四季都有风景用来发呆
# 我可以保证的是，任何经过“长时间”学习的人，内心绝对强大。因为，你能屏蔽暂时的诱惑，忍受内心的孤独。
# 命运不眷顾这样的人，命运都不好意思。
#
# 热水杯，大家的标配吧
# 可视化进步，提高成就感！螺旋上升，回顾总结！
# 做个进度跟踪表格，可视化自己学习过程。
#
# 跟踪学习进度的贴纸
# 笔记不是用来吃灰的！
# 你的“笔记”可能是可视化学习积累的最好“证据”。
#
# 某一张思维导图
# 长时间学习 = 投资时间！
# 当我们没钱的时候，最充足的资本就是时间！
# 有人“因看见而相信”，而你选择“因相信而看见！”你相信投入必然有回报！
#
# 从鸡兔同笼问题，到线性代数中的线性映射
#
# 身份证号21010419940529097
#
# 这是成书草稿中的“鸡兔互变”。从古老的鸡兔同笼问题，到线性代数，到回归，再到马尔可夫。
# 书都是人写的，书里面错误绝对不少。你的老师也会犯错，也未必能把某个问题理解透彻。因此，“我行，我上”也含有一层批判性学习的内涵！
# 其实，“鸢尾花书”这个系列七本书，就是源自“换我来讲，我会怎么讲”。
# 从中学，到大学，到读博士，我一直反感目前的数学教学方法。多说一嘴，数学 不是 习题集！数学 不是 解题技巧！数学 不是 考试！数学 不是 分数！数学是改天换地的刀枪剑戟！
# 与其抱怨，不如我来试试！结果，挖了个大坑，自己现在还在慢慢地填坑（长时间学习）...
#
# 图书馆一角六月份给我的惊喜。又一个年复一年如期而至的期待
# """  # 替换为实际文本
#
#     problem_zones = classifier.trace(long_text)
#     # # 输出问题上下文
#     print("\n问题上下文区域:")
#     for zone in problem_zones:
#         if zone['label'] != 'normal' and zone['score'] > 0.9:
#             print(
#                 f"\n在位置{zone['start_pos']}-{zone['end_pos']}检测到 [{zone['label']}] (置信度: {zone['score']:.2%})")
#             # print("上下文内容:", zone['context'][:100] + "...")
#             print("上下文内容:", zone['context'][:100])
#
