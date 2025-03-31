# 학습 과정과 추론 결과를 기록하기 위해 로깅
import logging
import os

def setup_logger(log_file="training.log"):
    logger = logging.getLogger("VisionLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # 중복 핸들러 방지
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger