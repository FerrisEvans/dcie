# from conf import database_conf, database_active
# from log import logging
# from typing import Optional
# from urllib.parse import quote_plus
# from sqlalchemy import *
# from sqlalchemy.orm import *
#
# import datetime
#
# engine = create_engine(f'postgresql+psycopg2://{database_conf["username"]}:{quote_plus(database_conf["password"])}@{database_conf["host"]}:{database_conf["port"]}/{database_conf["db"]}', echo=True)
#
# class BaseModel(DeclarativeBase):
#     id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
#     delete: Mapped[bool] = mapped_column(default=False)
#     created_time: Mapped[datetime.datetime] = mapped_column(default=lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#
# class SensitiveWord(BaseModel):
#     __tablename__ = "res_sensitive_word"
#     task_id: Mapped[str] # task id
#     filename: Mapped[str] # 文件名
#     res: Mapped[Optional[str]] # 扫描结果
#     res_ac: Mapped[Optional[str]] # AC自动机识别结果
#     toxic: Mapped[bool] # 是否有毒
#
#
# class Bert(BaseModel):
#     __tablename__ = "res_bert"
#     task_id: Mapped[str]
#     filename: Mapped[str]
#     label: Mapped[Optional[str]] # 标签
#     score: Mapped[Optional[float]] # 评分
#     start_pos: Mapped[Optional[int]] # 上下文开始位置
#     end_pos: Mapped[Optional[int]] # 上下文结束位置
#     context: Mapped[Optional[str]] # 上下文
#     toxic: Mapped[bool] # 是否有毒
#
#
# class Vocabulary(BaseModel):
#     __tablename__ = "vocabulary"
#     version: Mapped[int]
#     label: Mapped[str]
#     word: Mapped[str]
#
# class dataset(BaseModel):
#     __tablename__="dataset"
#     text: Mapped[str]
#     file_type: Mapped[str]
#     risk_level: Mapped[str]
#     labels: Mapped[str]
#     basis: Mapped[str]
#
# def sensitive_word(arr: list[SensitiveWord]):
#     if (not database_active) or (not arr):
#         return
#     try:
#         with Session(engine) as session:
#             session.bulk_save_objects(arr)
#             session.commit()
#     except Exception as e:
#         logging.error(f"Error: Detailed error: {e}")
#
#
#
# def bert(arr: list[Bert]):
#     if (not database_active) or (not arr):
#         return
#     try:
#         with Session(engine) as session:
#             session.bulk_save_objects(arr)
#             session.commit()
#     except Exception as e:
#         logging.error(f"Error: Detailed error: {e}")
#
