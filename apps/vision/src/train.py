# 데이터셋에서 라벨을 가져오도록 수정
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.model import VisionModel
from data.dataset import VisionDataset
from utils.preprocessing import get_transform
import yaml
import os
import numpy as np


def train(config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = VisionModel(num_classes=config['num_classes']).to(device)

  # 데이터셋 분할 (80% 학습, 20% 검증)
  transform = get_transform()
  dataset = VisionDataset(config['data_dir'], transform=transform)
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(0.2 * dataset_size))  # 20% 검증용
  np.random.shuffle(indices)
  train_indices, val_indices = indices[split:], indices[:split]

  train_sampler = SubsetRandomSampler(train_indices)
  val_sampler = SubsetRandomSampler(val_indices)

  train_loader = DataLoader(dataset, batch_size=config['batch_size'],
                            sampler=train_sampler)
  val_loader = DataLoader(dataset, batch_size=config['batch_size'],
                          sampler=val_sampler)

  optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
  criterion = torch.nn.CrossEntropyLoss()

  # 학습 및 검증 루프
  for epoch in range(config['epochs']):
    # 학습
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
      images, targets = images.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # 검증
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
      for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(
      f"Epoch [{epoch + 1}/{config['epochs']}] - Train Loss: {avg_train_loss:.4f}, "
      f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

  # 모델 저장
  os.makedirs("checkpoints", exist_ok=True)
  torch.save(model.state_dict(), "checkpoints/vision_model.pth")


if __name__ == "__main__":
  with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)
  train(config)