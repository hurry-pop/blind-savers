import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetModel(nn.Module):
  """
  EfficientNet을 기반으로 한 전이 학습 모델
  """

  def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
    super(EfficientNetModel, self).__init__()

    # EfficientNet-B0 로드
    self.model = models.efficientnet_b0(pretrained=pretrained)

    # 백본 동결 여부
    if freeze_backbone:
      for param in self.model.parameters():
        param.requires_grad = False

    # 분류기 부분 수정
    in_features = self.model.classifier[1].in_features
    self.model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )

  def forward(self, x):
    return self.model(x)


class MobileNetModel(nn.Module):
  """
  MobileNetV3를 기반으로 한 경량 모델
  """

  def __init__(self, num_classes=10, pretrained=True, freeze_backbone=True):
    super(MobileNetModel, self).__init__()

    # MobileNetV3-Small 로드
    self.model = models.mobilenet_v3_small(pretrained=pretrained)

    # 백본 동결 여부
    if freeze_backbone:
      for param in self.model.parameters():
        param.requires_grad = False

    # 분류기 부분 수정
    in_features = self.model.classifier[3].in_features
    self.model.classifier[3] = nn.Linear(in_features, num_classes)

  def forward(self, x):
    return self.model(x)


def get_model(config):
  """
  설정에 따라 적절한 모델 반환
  """
  num_classes = config['data']['num_classes']
  model_type = config['model']['type']
  pretrained = config['model']['pretrained']
  freeze_backbone = config['model']['freeze_backbone']

  if model_type == 'efficientnet':
    return EfficientNetModel(num_classes, pretrained, freeze_backbone)
  elif model_type == 'mobilenet':
    return MobileNetModel(num_classes, pretrained, freeze_backbone)
  elif model_type == 'custom':
    # 기존 코드의 VisionModel 사용
    from .model import VisionModel
    return VisionModel()
  else:
    raise ValueError(f"Unsupported model type: {model_type}")


# 객체 탐지 모델
class ObjectDetectionModel:
  """
  Faster R-CNN 기반 객체 탐지 모델
  """

  def __init__(self, num_classes, pretrained=True):
    self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # 클래스 수 조정 (배경 클래스 포함)
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes + 1
    )

  def forward(self, images, targets=None):
    return self.model(images, targets)

  def to(self, device):
    self.model = self.model.to(device)
    return self