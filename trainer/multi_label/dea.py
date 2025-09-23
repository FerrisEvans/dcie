import pandas as pd
import random
import os

# ========= 1. 标签体系 =========
parent_labels = ["身份信息", "汽车信息"]
child_labels = {
    "身份信息": ["身份证", "手机号", "家庭地址"],
    "汽车信息": ["车牌号", "车架号", "发动机型号"]
}

# ========= 2. 原始单标签样本 =========
single_samples = [
    {"text": "身份证：46040219501215641X", "labels": "身份信息,身份证"},
    {"text": "手机号：13800138000", "labels": "身份信息,手机号"},
    {"text": "家庭住址：北京市海淀区XX小区", "labels": "身份信息,家庭地址"},
    {"text": "车牌号：京A12345", "labels": "汽车信息,车牌号"},
    {"text": "车架号：LHG1234567890ABCDE", "labels": "汽车信息,车架号"},
    {"text": "发动机型号：JLH-4G20TDJ", "labels": "汽车信息,发动机型号"}
]

df_single = pd.DataFrame(single_samples)

# ========= 3. 典型组合样本 =========
typical_combos = [
    {"text": "身份证：46040219501215641X；手机号：13800138000",
     "labels": "身份信息,身份证,身份信息,手机号"},
    {"text": "车牌号：京A12345；发动机型号：JLH-4G20TDJ",
     "labels": "汽车信息,车牌号,汽车信息,发动机型号"},
    {"text": "身份证：46040219501215641X；车牌号：京A12345；发动机型号：JLH-4G20TDJ",
     "labels": "身份信息,身份证,汽车信息,车牌号,汽车信息,发动机型号"}
]

df_combos = pd.DataFrame(typical_combos)

# ========= 4. 数据增强：随机拼接单标签样本 =========
augmented_samples = []
num_augmented = 50  # 生成多少条增强样本

for _ in range(num_augmented):
    sample1 = df_single.sample(1).iloc[0]
    sample2 = df_single.sample(1).iloc[0]
    if sample1["labels"] != sample2["labels"]:
        new_text = sample1["text"] + "；" + sample2["text"]
        new_labels_set = set(sample1["labels"].split(",") + sample2["labels"].split(","))
        new_labels = ",".join(new_labels_set)
        augmented_samples.append({"text": new_text, "labels": new_labels})

df_augmented = pd.DataFrame(augmented_samples)

# ========= 5. 合并所有训练样本 =========
df_train = pd.concat([df_single, df_combos, df_augmented]).reset_index(drop=True)

# 打乱顺序
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# ========= 6. 保存到 CSV =========
save_dir = "train_data"
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "train_hierbert.csv")
df_train.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"✅ 训练数据已生成，路径：{csv_path}")
print(df_train.head(10))