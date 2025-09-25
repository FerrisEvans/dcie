from paddleocr import PaddleOCR

import threading
import os
import zipfile
import shutil


class Ocr:
    _instance = None  # 单例实例
    _lock = threading.Lock()  # 线程安全锁
    _initialized = False  # 防止 __init__ 重复执行

    def __new__(cls):
        """单例模式：确保全局唯一实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """need to run only once to download and load model into memory"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.img_model = PaddleOCR(use_doc_orientation_classify=False,
                                               use_doc_unwarping=False,
                                               use_textline_orientation=False,
                                               lang="ch")
                    # self.pdf_model = PaddleOCR(use_doc_orientation_classify=False,
                    #                            use_doc_unwarping=False,
                    #                            use_textline_orientation=False,
                    #                            ocr_version="PP-OCRv3",
                    #                            lang="ch")
                    self._initialized = True

    def read(self, path: str) -> str:
        text = ""
        result = self.img_model.predict(path)
        for res in result:
            # texts = [text for text in res["rec_texts"]]
            for t in res["rec_texts"]:
                text += t + "\n"

        print(text)
        return text

    def detailed_read(self, path: str) -> list:
        ret = []
        # 先用img model 做整体识别，包括图片和pdf
        result = self.img_model.predict(path)
        for res in result:
            formatted_data = {
                "input_path": res["input_path"],
                "res": [
                    {"text": text, "score": score}
                    for text, score in zip(res["rec_texts"], res["rec_scores"])]
            }
            ret.append(formatted_data)
        return ret

    def read_office_pic(self, file_path: str, output_folder: str = "/home/ecs-user/workspace/hai/data/res/"):
        """
        从word/ppt/xls中提取所有图片并保存到指定文件夹

        参数:
            docx_path: Word文档路径
            output_folder: 图片输出文件夹
        """
        text = ""
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        pic_dir = ""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".docx", ".doc"):
            pic_dir = "word"
        elif ext in ('.pptx', '.ppt'):
            pic_dir = "ppt"
        elif ext in ('.xlsx', '.xls'):
            pic_dir = "xl"
        else:
            return text

        # 临时解压目录
        temp_dir = os.path.join(output_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # 将.docx文件作为zip解压
            with zipfile.ZipFile(file_path) as docx_zip:
                docx_zip.extractall(temp_dir)

            # 图片存储在word/media目录下
            media_dir = os.path.join(temp_dir, pic_dir, "media")

            if os.path.exists(media_dir):
                # 复制所有图片到输出目录

                print("media_dir: ", media_dir)
                for img_file in os.listdir(media_dir):
                    text += self.read(f"{media_dir}/{img_file}") + "\n"
            return text
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    o = Ocr()
    img_path = "/home/ecs-user/workspace/hai/data/test.png"
    pdf_path = "/home/ecs-user/workspace/hai/data/2pdf.pdf"
    o.read(pdf_path)

    dest = "/home/ecs-user/workspace/hai/data/res/"
    doc = "/home/ecs-user/workspace/hai/data/test.docx"
    # print(o.read_pic_from_doc(doc, dest))


