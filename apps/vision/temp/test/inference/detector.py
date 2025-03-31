import torch
import numpy as np
import cv2
import yaml
import time
from pathlib import Path
import logging

# 프로젝트 모듈 임포트
from models.pretrained import get_model
from utils.preprocessing import preprocess_image, get_transform


class ObjectDetector:
  """
  실시간 객체 감지 클래스
  """

  def __init__(self, config_path="configs/config.yaml"):
    """
    매개변수:
        config_path (str): 설정 파일 경로
    """
    # 설정 로드
    with open(config_path, 'r') as f:
      self.config = yaml.safe_load(f)

    # 로깅 설정
    self.logger = logging.getLogger("blind_savers.detector")

    # 장치 설정
    self.device = torch.device(self.config['inference']['device'])
    self.logger.info(f"추론 장치: {self.device}")

    # 모델 로드
    self.model = self._load_model()
    self.model.eval()

    # 변환 설정
    self.transform = get_transform(self.config, is_train=False)

    # 클래스 이름
    self.class_names = self.config['data']['class_names']

    # 신뢰도 임계값
    self.confidence_threshold = self.config['inference']['confidence_threshold']

  def _load_model(self):
    """모델 로드 및 설정"""
    model_path = self.config['inference']['model_path']

    if not Path(model_path).exists():
      self.logger.warning(f"모델 파일이 존재하지 않습니다: {model_path}. 새 모델을 초기화합니다.")
      model = get_model(self.config)
    else:
      self.logger.info(f"모델 로드 중: {model_path}")
      checkpoint = torch.load(model_path, map_location=self.device)