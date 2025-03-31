import cv2
import numpy as np
from torchvision import transforms
from PIL import Image


def get_transform(config=None, is_train=True):
  """
    이미지 변환 설정을 반환

    매개변수:
        config (dict, optional): 설정 정보
        is_train (bool): 학습용 변환인지 여부

    반환:
        torchvision.transforms.Compose: 변환 컴포즈 객체
    """
  # 기본 설정
  if config is None:
    img_size = (64, 64)
    use_augmentation = False
  else:
    img_size = config.get('data', {}).get('img_size', (224, 224))
    use_augmentation = config.get('augmentation', {}).get('use_augmentation',
                                                          False)

  # 기본 변환 설정
  base_transforms = [
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]

  # 학습 모드이고 증강 설정이 활성화되어 있으면 데이터 증강 적용
  if is_train and use_augmentation and config:
    aug_config = config.get('augmentation', {})
    aug_transforms = []

    # 각종 증강 설정 적용
    if aug_config.get('rotation_range'):
      aug_transforms.append(
        transforms.RandomRotation(aug_config.get('rotation_range')))

    if aug_config.get('flip_horizontal', False):
      aug_transforms.append(transforms.RandomHorizontalFlip())

    if aug_config.get('flip_vertical', False):
      aug_transforms.append(transforms.RandomVerticalFlip())

    if aug_config.get('random_crop', False):
      aug_transforms.append(
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))

    # 밝기, 대비, 채도 등 조정
    if any(
        [aug_config.get('brightness_range'), aug_config.get('contrast_range'),
         aug_config.get('saturation_range')]):
      color_jitter_params = {}

      if aug_config.get('brightness_range'):
        brightness = aug_config.get('brightness_range')
        color_jitter_params['brightness'] = (min(brightness), max(brightness))

      aug_transforms.append(transforms.ColorJitter(**color_jitter_params))

    # 증강 변환과 기본 변환 결합
    transform_list = aug_transforms + base_transforms
  else:
    transform_list = base_transforms

  return transforms.Compose(transform_list)


def preprocess_image(image, target_size=None):
  """
    이미지 전처리 함수

    매개변수:
        image (str or numpy.ndarray): 이미지 경로 또는 이미지 배열
        target_size (tuple, optional): 조정할 이미지 크기 (높이, 너비)

    반환:
        numpy.ndarray: 전처리된 이미지
    """
  # 이미지 로드
  if isinstance(image, str):
    image = cv2.imread(image)
    if image is None:
      raise ValueError(f"이미지를 로드할 수 없습니다: {image}")

  # BGR -> RGB 변환
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # 크기 조정 (필요한 경우)
  if target_size is not None:
    image = cv2.resize(image, (target_size[1], target_size[0]))

  return image


def prepare_for_detection(image, transform=None, device='cpu'):
  """
    객체 감지를 위한 이미지 준비

    매개변수:
        image (numpy.ndarray): 원본 이미지
        transform (callable, optional): 적용할 변환
        device (str): 텐서를 저장할 장치

    반환:
        torch.Tensor: 모델 입력용 텐서
    """
  import torch

  # PIL 이미지로 변환
  if isinstance(image, np.ndarray):
    image = Image.fromarray(image)

  # 변환 적용
  if transform:
    image = transform(image)

  # 배치 차원 추가
  image = image.unsqueeze(0)

  # 장치 이동
  image = image.to(device)

  return image


def convert_to_heatmap(confidence_map, original_image, alpha=0.7):
  """
    신뢰도 맵을 히트맵으로 변환하여 원본 이미지에 오버레이

    매개변수:
        confidence_map (numpy.ndarray): 신뢰도 맵
        original_image (numpy.ndarray): 원본 이미지
        alpha (float): 블렌딩 강도

    반환:
        numpy.ndarray: 히트맵이 오버레이된 이미지
    """
  # 스케일링 및 히트맵 변환
  confidence_map = (confidence_map - confidence_map.min()) / (
        confidence_map.max() - confidence_map.min() + 1e-8)
  confidence_map = (confidence_map * 255).astype(np.uint8)
  heatmap = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)

  # 히트맵 크기 조정
  heatmap = cv2.resize(heatmap,
                       (original_image.shape[1], original_image.shape[0]))

  # 원본 이미지에 오버레이
  overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

  return overlay