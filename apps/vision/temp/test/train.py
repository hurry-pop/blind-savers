import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import yaml
import os
import time
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 프로젝트 모듈 임포트
from models.pretrained import get_model
from data.dataset import VisionDataset
from utils.preprocessing import get_transform
from utils.evaluation import evaluate_model, compute_confusion_matrix


def setup_logging(config):
  """로깅 설정"""
  log_dir = Path(config['logging']['log_dir'])
  log_dir.mkdir(exist_ok=True, parents=True)

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      handlers=[
        logging.FileHandler(log_dir / 'training.log'),
        logging.StreamHandler()
      ]
  )

  logger = logging.getLogger('blind_savers')

  if config['logging']['tensorboard']:
    writer = SummaryWriter(log_dir / 'tensorboard')
  else:
    writer = None

  return logger, writer


def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
  """체크포인트 저장"""
  checkpoint_dir = Path(config['evaluation']['checkpoint_dir'])
  checkpoint_dir.mkdir(exist_ok=True, parents=True)

  checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': metrics,
    'config': config
  }

  # 일반 체크포인트 저장
  checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
  torch.save(checkpoint, checkpoint_path)

  # 최고 성능 모델 저장
  if is_best:
    best_model_path = checkpoint_dir / "best_model.pth"
    torch.save(checkpoint, best_model_path)

  return checkpoint_path


def train(config):
  """모델 학습 함수"""
  # 로깅 설정
  logger, writer = setup_logging(config)
  logger.info(f"학습 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  logger.info(f"설정: {config}")

  # 장치 설정
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f"사용 장치: {device}")

  # 변환 설정
  train_transform = get_transform(config, is_train=True)
  val_transform = get_transform(config, is_train=False)

  # 데이터셋 및 데이터로더 설정
  train_dataset = VisionDataset(
      data_dir=config['data']['train_dir'],
      transform=train_transform,
      split='train'
  )

  val_dataset = VisionDataset(
      data_dir=config['data']['val_dir'],
      transform=val_transform,
      split='val'
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=config['training']['batch_size'],
      shuffle=True,
      num_workers=config['training']['num_workers'],
      pin_memory=True
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=config['training']['batch_size'],
      shuffle=False,
      num_workers=config['training']['num_workers'],
      pin_memory=True
  )

  logger.info(f"학습 데이터셋 크기: {len(train_dataset)}")
  logger.info(f"검증 데이터셋 크기: {len(val_dataset)}")

  # 클래스 분포 로깅
  train_distribution = train_dataset.get_class_distribution()
  logger.info(f"학습 데이터 클래스 분포: {train_distribution}")

  # 모델 설정
  model = get_model(config)
  model = model.to(device)

  # 손실 함수 및 옵티마이저 설정
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(
      model.parameters(),
      lr=config['training']['learning_rate'],
      weight_decay=config['training']['weight_decay']
  )

  # 학습률 스케줄러
  scheduler = StepLR(
      optimizer,
      step_size=config['training']['scheduler_step_size'],
      gamma=config['training']['scheduler_gamma']
  )

  # 조기 종료 설정
  early_stopping_patience = config['training']['early_stopping_patience']
  best_val_loss = float('inf')
  no_improve_epochs = 0

  # 학습 루프
  num_epochs = config['training']['epochs']
  log_interval = config['logging']['log_interval']

  for epoch in range(num_epochs):
    # 학습 모드
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_pbar = tqdm(train_loader,
                      desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

    for batch_idx, (data, targets) in enumerate(train_pbar):
      data, targets = data.to(device), targets.to(device)

      # 그래디언트 초기화
      optimizer.zero_grad()

      # 순전파
      outputs = model(data)
      loss = criterion(outputs, targets)

      # 역전파
      loss.backward()
      optimizer.step()

      # 통계 업데이트
      train_loss += loss.item()
      _, predicted = outputs.max(1)
      train_total += targets.size(0)
      train_correct += predicted.eq(targets).sum().item()

      # 진행 상황 업데이트
      train_pbar.set_postfix({
        'loss': train_loss / (batch_idx + 1),
        'acc': 100. * train_correct / train_total
      })

      # 로깅
      if batch_idx % log_interval == 0:
        if writer:
          step = epoch * len(train_loader) + batch_idx
          writer.add_scalar('Loss/train_step', loss.item(), step)

    # 에폭 평균 손실 및 정확도
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total

    # 검증 단계
    val_loss, val_metrics = evaluate_model(
        model, val_loader, criterion, device, config['data']['num_classes']
    )

    # 스케줄러 단계
    scheduler.step()

    # 로깅
    logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")

    if writer:
      writer.add_scalar('Loss/train', train_loss, epoch)
      writer.add_scalar('Loss/val', val_loss, epoch)
      writer.add_scalar('Accuracy/train', train_acc, epoch)
      writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)

      # 혼동 행렬 계산 및 저장 (5 에폭마다)
      if (epoch + 1) % 5 == 0 and config['evaluation']['confusion_matrix']:
        cm = compute_confusion_matrix(model, val_loader, device,
                                      config['data']['num_classes'])
        # 혼동 행렬 이미지 로깅 로직은 별도로 구현 필요

    # 체크포인트 저장
    is_best = val_loss < best_val_loss
    if is_best:
      best_val_loss = val_loss
      no_improve_epochs = 0
    else:
      no_improve_epochs += 1

    save_checkpoint(
        model, optimizer, epoch, val_metrics, config, is_best
    )

    # 조기 종료 체크
    if no_improve_epochs >= early_stopping_patience:
      logger.info(f"조기 종료: {early_stopping_patience}에폭 동안 성능 개선 없음")
      break

  # 학습 완료
  logger.info(f"학습 완료: {time.strftime('%Y-%m-%d %H:%M:%S')}")

  if writer:
    writer.close()

  return model


if __name__ == "__main__":
  # 설정 파일 로드
  with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

  # 학습 실행
  trained_model = train(config)