import torch
import numpy as np
import cv2
import yaml
import time
from pathlib import Path
import logging
from PIL import Image
import os

# 프로젝트 모듈 임포트
from models.pretrained import get_model
from utils.preprocessing import preprocess_image, prepare_for_detection, \
  get_transform


class ObjectClassifier:
  """
  이미지 분류 클래스
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
    self.logger = logging.getLogger("blind_savers.classifier")

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
      model = get_model(self.config)
      model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(self.device)
    return model

  def predict(self, image_source):
    """
    이미지 분류 예측

    매개변수:
        image_source (str or numpy.ndarray): 이미지 경로 또는 이미지 배열

    반환:
        dict: 예측 결과
    """
    # 이미지 로드 및 전처리
    if isinstance(image_source, str):
      if not os.path.exists(image_source):
        self.logger.error(f"이미지 파일이 존재하지 않습니다: {image_source}")
        return {
          'success': False,
          'error': f"Image file not found: {image_source}"
        }

      image = Image.open(image_source).convert('RGB')
    elif isinstance(image_source, np.ndarray):
      image = Image.fromarray(image_source)
    else:
      self.logger.error(f"지원되지 않는 이미지 소스 형식: {type(image_source)}")
      return {
        'success': False,
        'error': f"Unsupported image source type: {type(image_source)}"
      }

    # 변환 적용
    if self.transform:
      transformed_image = self.transform(image)
    else:
      transformed_image = get_transform()(image)

    # 배치 차원 추가
    input_tensor = transformed_image.unsqueeze(0).to(self.device)

    # 예측 수행
    start_time = time.time()

    try:
      with torch.no_grad():
        outputs = self.model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

      # 결과 처리
      probs, indices = torch.topk(probabilities,
                                  k=min(3, len(self.class_names)))
      probs = probs.squeeze().cpu().numpy()
      indices = indices.squeeze().cpu().numpy()

      inference_time = time.time() - start_time

      # 결과 포맷팅
      predictions = [
        {
          'class_id': int(idx),
          'class_name': self.class_names[idx],
          'confidence': float(prob)
        }
        for idx, prob in zip(indices, probs)
        if prob >= self.confidence_threshold
      ]

      return {
        'success': True,
        'predictions': predictions,
        'inference_time': inference_time
      }

    except Exception as e:
      self.logger.error(f"예측 중 오류 발생: {str(e)}")
      return {
        'success': False,
        'error': str(e)
      }

  def process_image_file(self, image_path):
    """
    이미지 파일 처리 및 결과 반환

    매개변수:
        image_path (str): 이미지 파일 경로

    반환:
        dict: 예측 결과
    """
    # 이미지 예측
    result = self.predict(image_path)

    if not result['success']:
      return result

    # 원본 이미지 로드 (시각화용)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 결과 시각화 (구현 필요)

    return result

  def process_camera_frame(self, frame):
    """
    카메라 프레임 처리 및 결과 반환

    매개변수:
        frame (numpy.ndarray): 카메라 프레임

    반환:
        dict: 예측 결과
    """
    # 이미지 예측
    result = self.predict(frame)

    if not result['success']:
      return result

    # 결과 시각화 (구현 필요)

    return result

  def generate_audio_feedback(self, result):
    """
    예측 결과를 기반으로 오디오 피드백 생성

    매개변수:
        result (dict): 예측 결과

    반환:
        str: 음성 피드백 텍스트
    """
    if not result['success'] or not result['predictions']:
      return "객체를 인식할 수 없습니다. 다시 시도해주세요."

    top_prediction = result['predictions'][0]
    class_name = top_prediction['class_name']
    confidence = top_prediction['confidence']

    # 신뢰도에 따른 메시지 조정
    if confidence >= 0.9:
      confidence_message = "매우 높은 확률로"
    elif confidence >= 0.7:
      confidence_message = "높은 확률로"
    elif confidence >= 0.5:
      confidence_message = "중간 확률로"
    else:
      confidence_message = "낮은 확률로"

    message = f"{confidence_message} {class_name}입니다."

    # 클래스별 추가 정보
    if class_name == "tactile_paving":
      message += " 점자블록이 감지되었습니다. 보행 경로를 따라가세요."
    elif class_name == "obstacle":
      message += " 장애물이 감지되었습니다. 주의하세요."
    elif class_name == "stairs":
      message += " 계단이 감지되었습니다. 천천히 조심해서 이동하세요."
    elif class_name == "crosswalk":
      message += " 횡단보도가 감지되었습니다. 신호를 확인하세요."

    return message