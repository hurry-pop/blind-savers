import torch
import torch.nn as nn
import torchvision.models as models


class VisionModel(nn.Module):
  def __init__(self, num_classes):
    super(VisionModel, self).__init__()
    # 사전 학습된 MobileNetV2 사용
    self.backbone = models.mobilenet_v2(pretrained=True)
    # 마지막 분류 레이어 교체
    self.backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)  # MobileNetV2의 출력 크기는 1280
    )

  def forward(self, x):
    return self.backbone(x)


# 간단한 CNN을 선호한다면:
class SimpleVisionModel(nn.Module):
  def __init__(self, num_classes):
    super(SimpleVisionModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.fc1 = nn.Linear(128 * 16 * 16, 256)  # 64x64 입력 기준
    self.fc2 = nn.Linear(256, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x