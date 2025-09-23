import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import os

# ========= 配置 =========
MODEL_NAME = "hfl/chinese-bert-wwm-ext"   # 中文 BERT
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

# ========= 1. 定义标签体系 =========

parent_labels = ["身份信息", "汽车信息"]
child_labels = {
    "身份信息": ["身份证", "手机号", "家庭地址"],
    "汽车信息": ["车牌号", "车架号", "发动机型号"]
}

# 映射到 ID
parent2id = {l: i for i, l in enumerate(parent_labels)}
id2parent = {i: l for l, i in parent2id.items()}

child2id = {}
id2child = {}
for p, subs in child_labels.items():
    for s in subs:
        cid = len(child2id)
        child2id[s] = cid
        id2child[cid] = s

# ========= 2. 数据集 =========
class HierDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_list = self.labels[idx].split(",")

        # 父标签 one-hot
        parent_targets = torch.zeros(len(parent2id))
        for l in label_list:
            if l in parent2id:
                parent_targets[parent2id[l]] = 1

        # 子标签 one-hot
        child_targets = torch.zeros(len(child2id))
        for l in label_list:
            if l in child2id:
                child_targets[child2id[l]] = 1

        encodings = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["parent_labels"] = parent_targets
        item["child_labels"] = child_targets
        return item

# 加载模型类（要和训练时保持一致）



# ========= 3. 模型 =========
class HierBERT(nn.Module):
    def __init__(self, model_name, num_parent, num_child):
        super(HierBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.parent_classifier = nn.Linear(hidden_size, num_parent)
        self.child_classifier = nn.Linear(hidden_size + num_parent, num_child)  # 拼接父标签预测作为子标签输入

    def forward(self, input_ids, attention_mask, parent_labels=None, child_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden]  # [CLS] 向量
        parent_logits = self.parent_classifier(pooled)  # [batch, num_parent]
        # 将父预测拼接进子分类器
        parent_probs = torch.sigmoid(parent_logits)
        combined = torch.cat([pooled, parent_probs], dim=1)
        child_logits = self.child_classifier(combined)  # [batch, num_child]
        loss = None
        if parent_labels is not None and child_labels is not None:
            bce = nn.BCEWithLogitsLoss()
            parent_loss = bce(parent_logits, parent_labels)
            child_loss = bce(child_logits, child_labels)
            loss = parent_loss + child_loss
        return parent_logits, child_logits, loss  # 注意：返回 logits（未 sigmoid）

# ========= 4. 训练准备 =========
df = pd.read_csv("/test/data.csv")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = HierDataset(train_df, tokenizer, MAX_LEN)
val_dataset = HierDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = HierBERT(MODEL_NAME, len(parent2id), len(child2id))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)

# ========= 5. 训练 & 验证 =========
from sklearn.metrics import f1_score

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        parent_labels = batch["parent_labels"].to(device)
        child_labels = batch["child_labels"].to(device)

        # parent_logits, child_logits, loss = model.forward(input_ids, attention_mask, parent_labels, child_labels)
        # forward 要 tensor
        parent_logits, child_logits, loss = model(input_ids, attention_mask, parent_labels, child_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader):
    model.eval()
    all_parent_preds, all_parent_trues = [], []
    all_child_preds, all_child_trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            parent_labels = batch["parent_labels"].to(device)
            child_labels = batch["child_labels"].to(device)

            # parent_logits, child_logits, loss = model.forward(input_ids, attention_mask, parent_labels, child_labels)
            # forward 要 tensor
            parent_logits, child_logits, loss = model(input_ids, attention_mask, parent_labels, child_labels)

            # logits → prob → numpy
            parent_probs = torch.sigmoid(parent_logits).cpu().numpy()
            child_probs = torch.sigmoid(child_logits).cpu().numpy()

            # label 也转成 numpy 用来算 f1
            parent_labels = parent_labels.cpu().numpy()
            child_labels = child_labels.cpu().numpy()

            parent_preds = (parent_probs > 0.5).astype(int)
            child_preds = (child_probs > 0.5).astype(int)

            all_parent_preds.extend(parent_preds)
            all_parent_trues.extend(parent_labels)
            all_child_preds.extend(child_preds)
            all_child_trues.extend(child_labels)

    # micro-F1（总体）
    parent_f1 = f1_score(all_parent_trues, all_parent_preds, average="micro", zero_division=0)
    child_f1 = f1_score(all_child_trues, all_child_preds, average="micro", zero_division=0)

    print("\nParent Report:")
    print(classification_report(all_parent_trues, all_parent_preds,
                                target_names=list(parent2id.keys()), zero_division=0))

    print("\nChild Report:")
    print(classification_report(all_child_trues, all_child_preds,
                                target_names=list(child2id.keys()), zero_division=0))

    return parent_f1, child_f1


def train1():
    # ========= 6. 主循环（新增保存逻辑） =========
    best_f1 = 0.0
    save_dir = "/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}")

        parent_f1, child_f1 = eval_model(model, val_loader)
        avg_f1 = (parent_f1 + child_f1) / 2
        print(f"Validation F1 - Parent: {parent_f1:.4f}, Child: {child_f1:.4f}, Avg: {avg_f1:.4f}")

        # 保存当前 epoch 模型
        epoch_path = os.path.join(save_dir, f"hierbert_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"✅ 模型已保存到 {epoch_path}")

        # 保存最佳模型
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_path = os.path.join(save_dir, "hierbert_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 最佳模型更新，已保存到 {best_path}")


# --- 加载模型权重（更稳健） ---
def load_model(model_path):
    model = HierBERT(MODEL_NAME, num_parent=len(parent2id), num_child=len(child2id))

    # 先加载到 CPU，之后再移动到 device（兼容 CPU/GPU）
    state = torch.load(model_path, map_location="cpu")

    # 如果保存的是 dict（最常见），进一步处理
    if isinstance(state, dict):
        # 有些保存会包一层，如 {'state_dict': {...}}
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        # 处理 DataParallel 保存时的 module. 前缀
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        state = new_state

        # 尝试严格匹配加载；若不匹配再降级为 strict=False 并打印警告
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f"[WARN] 严格加载失败：{e}\n尝试使用 strict=False 加载（可能有部分键不匹配）。")
            model.load_state_dict(state, strict=False)
    else:
        # 如果保存的是整个 model 对象（不常见），直接替换
        try:
            model = state
        except Exception as e:
            raise RuntimeError("无法加载模型权重：%s" % e)

    model.to(device)
    model.eval()
    return model


# --- 解析 model outputs 的工具函数（更鲁棒） ---
def parse_model_output(outputs):
    """
    尝试从 model(...) 的返回值中提取 parent_logits, child_logits（均为未 sigmoid 的 logits tensor）。
    支持多种形式的 outputs：dict / tuple / list / 包含 None 的 tuple 等。
    """
    parent_logits = None
    child_logits = None

    # 1) dict 情况（优先处理）
    if isinstance(outputs, dict):
        if "parent_logits" in outputs and "child_logits" in outputs:
            parent_logits = outputs["parent_logits"]
            child_logits = outputs["child_logits"]
        elif "parent_probs" in outputs and "child_probs" in outputs:
            # 将概率转回 logits（小心数值稳定性）
            p = outputs["parent_probs"].clamp(1e-6, 1 - 1e-6)
            c = outputs["child_probs"].clamp(1e-6, 1 - 1e-6)
            parent_logits = torch.log(p / (1 - p))
            child_logits = torch.log(c / (1 - c))
        elif "logits" in outputs:
            logits = outputs["logits"]
            if isinstance(logits, torch.Tensor) and logits.dim() == 2 and logits.size(1) == (len(parent2id) + len(child2id)):
                parent_logits = logits[:, :len(parent2id)]
                child_logits = logits[:, len(parent2id):]
        else:
            # 扫描字典中的 tensor，按列数匹配
            for v in outputs.values():
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    if v.size(1) == len(parent2id) and parent_logits is None:
                        parent_logits = v
                    elif v.size(1) == len(child2id) and child_logits is None:
                        child_logits = v

    # 2) tuple/list 情况
    elif isinstance(outputs, (tuple, list)):
        # 提取所有 tensor 项（忽略 None 或标量）
        tensors = [x for x in outputs if isinstance(x, torch.Tensor)]
        # 首先尝试按列数直接匹配（适合本例：(parent_logits, child_logits, None)）
        for t in tensors:
            if t.dim() == 2:
                if t.size(1) == len(parent2id) and parent_logits is None:
                    parent_logits = t
                elif t.size(1) == len(child2id) and child_logits is None:
                    child_logits = t

        # 如果还没找到，尝试在 tuples 中寻找形状最接近的 tensor（更宽松的回退）
        if parent_logits is None or child_logits is None:
            candidates = [t for t in tensors if t.dim() == 2]
            for t in candidates:
                if parent_logits is None and t.size(1) == len(parent2id):
                    parent_logits = t
                if child_logits is None and t.size(1) == len(child2id):
                    child_logits = t

    # 3) 失败时打印 debug 信息，便于定位
    if parent_logits is None or child_logits is None:
        # 打印一些有用信息帮助调试，然后抛错
        print("=== DEBUG: 无法解析 model 输出，outputs 内容如下： ===")
        print("Type:", type(outputs))
        try:
            # 可能是较大的 tensor，打印 summary 而非全部内容
            if isinstance(outputs, (tuple, list)):
                for i, o in enumerate(outputs):
                    print(f"  outputs[{i}]: type={type(o)}", end="")
                    if isinstance(o, torch.Tensor):
                        print(f", shape={tuple(o.shape)}, device={o.device}")
                    else:
                        print(f", value={o}")
            elif isinstance(outputs, dict):
                for k, v in outputs.items():
                    print(f"  outputs['{k}']: type={type(v)}", end="")
                    if isinstance(v, torch.Tensor):
                        print(f", shape={tuple(v.shape)}, device={v.device}")
                    else:
                        print(f", value={v}")
            else:
                print(outputs)
        except Exception:
            print("  (打印 outputs 时发生错误)")
        raise ValueError(
            "无法从 model 输出中提取 parent/child logits。请根据上方 debug 信息调整模型的 forward 返回值或修改 parse_model_output。"
        )

    return parent_logits, child_logits

# --- 预测函数（返回标签 + 概率） ---
def predict(text_or_texts, model, tokenizer, threshold=0.5):
    # 支持单条文本或多条文本列表
    single = False
    if isinstance(text_or_texts, str):
        texts = [text_or_texts]
        single = True
    else:
        texts = list(text_or_texts)

    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    parent_logits, child_logits = parse_model_output(outputs)
    parent_probs = torch.sigmoid(parent_logits).cpu().numpy()
    child_probs = torch.sigmoid(child_logits).cpu().numpy()

    results = []
    for i in range(parent_probs.shape[0]):
        p_probs = parent_probs[i]
        c_probs = child_probs[i]
        p_pred = [id2parent[idx] for idx, p in enumerate(p_probs) if p >= threshold]
        c_pred = [id2child[idx] for idx, p in enumerate(c_probs) if p >= threshold]
        results.append({
            "parent_pred": p_pred,
            "parent_probs": p_probs,
            "child_pred": c_pred,
            "child_probs": c_probs
        })

    return results[0] if single else results


def run(text: str):
    MODEL_PATH = "/saved_models/hierbert_best.pt"
    out = predict(text, load_model(MODEL_PATH), tokenizer, threshold=0.5)
    print("父标签预测：", out["parent_pred"])
    print("子标签预测：", out["child_pred"])
    print("父标签概率：", out["parent_probs"])
    print("子标签概率：", out["child_probs"])
    print(out)


if __name__ == "__main__":
    # train1()
    run("发动机型号：JLH-4G20TDJ；家庭住址：北京市海淀区XX小区")
    pass
