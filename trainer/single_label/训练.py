from datasets import DatasetDict, load_dataset, ClassLabel, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging
import torch
import os

# ===================== 可调参数配置 =====================
# 模型参数
MODEL_NAME = "distilbert-base-multilingual-cased"  # 预训练模型名称
PROBLEM_TYPE = "single_label_classification"       # 问题类型

# 数据参数
TRAIN_DATA_PATH = "./train-0922-1.xlsx"       # 训练数据路径
TEST_DATA_PATH = "./test-0922-1.xlsx"         # 测试数据路径
TEXT_COLUMN = "text"                              # 文本列名
LABEL_COLUMN = "labels"                           # 标签列名

# 训练参数
BATCH_SIZE = 32                                   # 批次大小
NUM_EPOCHS = 3                                    # 训练轮数
LEARNING_RATE = 3e-5                              # 学习率
WEIGHT_DECAY = 0.01                               # 权重衰减
MAX_LENGTH = 512                                  # 最大文本长度
GRADIENT_ACCUMULATION_STEPS = 2                   # 梯度累积步数

# 评估参数
EVAL_STRATEGY = "steps"                           # 评估策略
EVAL_STEPS = 100                                  # 评估步数
SAVE_STRATEGY = "steps"                           # 保存策略
SAVE_STEPS = 100                                  # 保存步数
LOGGING_STEPS = 50                                # 日志记录步数
METRIC_FOR_BEST_MODEL = "accuracy"                # 最佳模型指标

# 输出参数
OUTPUT_DIR = "./results"                          # 输出目录
LOGGING_DIR = "./logs"                            # 日志目录
SAVE_TOTAL_LIMIT = 2                              # 保存模型数量限制
BEST_MODEL_DIR = "best_model"                     # 最佳模型保存目录

# 设备参数
FP16 = True                                       # 是否使用混合精度训练
# ===================== 参数配置结束 =====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalTextClassifier:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_info = self.get_label_info()
        self.model = None
        self.dataset = None
        self.tokenized_dataset = None
        logger.info(f"所有标签类别: {self.label_info['names']}")

    def get_label_info(self):
        """从训练数据中获取标签元数据"""
        df = pd.read_excel(TEST_DATA_PATH)
        label_names = sorted(df[LABEL_COLUMN].unique().tolist())
        return {
            "names": label_names,
            "num_labels": len(label_names),
            "class_label": ClassLabel(names=label_names),
            "id2label": {i: label for i, label in enumerate(label_names)},
            "label2id": {label: i for i, label in enumerate(label_names)}
        }

    def load_data(self):
        """加载并预处理数据"""
        logger.info("Loading and preprocessing data...")

        def load_and_cast(path):
            df = pd.read_excel(path)
            ds = Dataset.from_pandas(df)
            return ds.cast_column(LABEL_COLUMN, self.label_info["class_label"])

        train_data = load_and_cast(TRAIN_DATA_PATH)
        test_data = load_and_cast(TEST_DATA_PATH)

        def tokenize_function(examples):
            texts = [
                str(text) if text is not None else "" for text in examples[TEXT_COLUMN]]
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length"
            )

        self.tokenized_dataset = DatasetDict({
            "train": train_data.map(tokenize_function, batched=True, num_proc=4),
            "test": test_data.map(tokenize_function, batched=True, num_proc=4)
        }).remove_columns([TEXT_COLUMN])

        logger.info("Data loading and preprocessing completed")

    def initialize_model(self):
        """初始化单标签分类模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.label_info["num_labels"],
            problem_type=PROBLEM_TYPE,
            id2label=self.label_info["id2label"],
            label2id=self.label_info["label2id"]
        )
        return self.model

    def get_training_args(self, output_dir: str = OUTPUT_DIR) -> TrainingArguments:
        """获取训练参数"""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=SAVE_STEPS,
            logging_dir=LOGGING_DIR,
            logging_steps=LOGGING_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model=METRIC_FOR_BEST_MODEL,
            greater_is_better=True,
            fp16=FP16,
            save_total_limit=SAVE_TOTAL_LIMIT,
            lr_scheduler_type="cosine_with_restarts",
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
        )

    def compute_metrics(self, p):
        """计算单标签分类评估指标"""
        preds = np.argmax(p.predictions, axis=1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1": f1_score(p.label_ids, preds, average="macro"),
            "precision": precision_score(p.label_ids, preds, average="macro"),
            "recall": recall_score(p.label_ids, preds, average="macro")
        }

    def train(self):
        """训练模型"""
        if self.model is None:
            self.initialize_model()

        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            compute_metrics=self.compute_metrics
        )

        logger.info("Starting training...")
        try:
            trainer.train()
            logger.info("Training completed successfully")
            return trainer
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def save_label_mapping(label_info, save_dir):
    """保存标签映射信息到单独的文件"""
    import json
    mapping_file = os.path.join(save_dir, "label_mapping.json")
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": label_info["id2label"],
            "label2id": label_info["label2id"],
            "label_names": label_info["names"]
        }, f, ensure_ascii=False, indent=2)


def main():
    classifier = LegalTextClassifier()
    classifier.load_data()
    trainer = classifier.train()

    # 确保输出目录存在
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # 保存模型和tokenizer
    trainer.save_model(BEST_MODEL_DIR)
    classifier.tokenizer.save_pretrained(BEST_MODEL_DIR)

    # 保存标签映射信息
    save_label_mapping(classifier.label_info, BEST_MODEL_DIR)

    logger.info(
        f"Model and label mapping saved to '{BEST_MODEL_DIR}' directory")


if __name__ == "__main__":
    main()

    model = AutoModelForSequenceClassification.from_pretrained("best_model")
    print(model.config.id2label)  # 现在应该显示实际的标签名称而不是数字
