from typing import Any, Optional

from sqlalchemy.orm import Mapped

from base import BaseModel

class Ac(BaseModel):
    __tablename__ = "res_ac"
    task_id: Mapped[str] # task id
    filename: Mapped[str] # 文件名
    res: Mapped[Optional[str]] # AC自动机识别结果


class Bert(BaseModel):
    __tablename__ = "res_bert"
    task_id: Mapped[str]
    filename: Mapped[str]
    label: Mapped[Optional[str]] # 标签
    score: Mapped[Optional[float]] # 评分
    start_pos: Mapped[Optional[int]] # 上下文开始位置
    end_pos: Mapped[Optional[int]] # 上下文结束位置
    context: Mapped[Optional[str]] # 上下文