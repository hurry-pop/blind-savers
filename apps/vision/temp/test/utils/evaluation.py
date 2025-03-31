import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
  f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger('blind_savers.evaluation')


def evaluate_model(model, data_loader, criterion, device, num_classes):
  """
  모델 평가

  매개변수:
      model: 평가할 모델
      data_loader: 데이터 로더
      criterion: 손실 함수
      device: 계산 장치
      num_classes: 클래스 수

  반환:
      float: 평균 손실
      dict: 평가 메트릭
  """
  model.eval()
  val_loss = 0.0
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for data, targets in tqdm(data_loader, desc="Evaluating"):
      data, targets = data.to(device), targets.to(device)

      outputs = model(data)
      loss = criterion(outputs, targets)

      val_loss += loss.item()

      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(targets.cpu().numpy())

  # 손실 계산
  val_loss /= len(data_loader)

  # 메트릭 계산
  accuracy = accuracy_score(all_targets, all_preds) * 100

  metrics = {
    'accuracy': accuracy
  }

  # 멀티클래스 메트릭 계산
  if num_classes > 2:
    metrics['precision'] = precision_score(all_targets, all_preds,
                                           average='macro') * 100
    metrics['recall'] = recall_score(all_targets, all_preds,
                                     average='macro') * 100
    metrics['f1'] = f1_score(all_targets, all_preds, average='macro') * 100
  else:
    metrics['precision'] = precision_score(all_targets, all_preds) * 100
    metrics['recall'] = recall_score(all_targets, all_preds) * 100
    metrics['f1'] = f1_score(all_targets, all_preds) * 100

  logger.info(f"평가 결과: 손실={val_loss:.4f}, 정확도={accuracy:.2f}%, "
              f"정밀도={metrics['precision']:.2f}%, 재현율={metrics['recall']:.2f}%, "
              f"F1={metrics['f1']:.2f}%")

  return val_loss, metrics


def compute_confusion_matrix(model, data_loader, device, num_classes,
    class_names=None):
  """
  혼동 행렬 계산

  매개변수:
      model: 평가할 모델
      data_loader: 데이터 로더
      device: 계산 장치
      num_classes: 클래스 수
      class_names: 클래스 이름 목록

  반환:
      numpy.ndarray: 혼동 행렬
  """
  model.eval()
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for data, targets in tqdm(data_loader, desc="Computing Confusion Matrix"):
      data, targets = data.to(device), targets.to(device)

      outputs = model(data)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_targets.extend(targets.cpu().numpy())

  # 혼동 행렬 계산
  cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))

  # 클래스 이름이 제공되지 않으면 숫자로 대체
  if class_names is None:
    class_names = [str(i) for i in range(num_classes)]

  return cm


def plot_confusion_matrix(cm, class_names, save_path=None):
  """
  혼동 행렬 시각화

  매개변수:
      cm: 혼동 행렬
      class_names: 클래스 이름 목록
      save_path: 저장 경로 (선택 사항)

  반환:
      matplotlib.figure.Figure: 그림 객체
  """
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
              yticklabels=class_names)
  plt.xlabel('예측 클래스')
  plt.ylabel('실제 클래스')
  plt.title('혼동 행렬')

  if save_path:
    plt.savefig(save_path, bbox_inches='tight')

  return plt.gcf()


def evaluate_object_detection(model, data_loader, device, iou_threshold=0.5):
  """
  객체 탐지 모델 평가

  매개변수:
      model: 평가할 모델
      data_loader: 데이터 로더
      device: 계산 장치
      iou_threshold: IoU 임계값

  반환:
      dict: mAP, 정밀도, 재현율 등의 평가 메트릭
  """
  model.eval()

  # 평가 결과 저장용 리스트
  all_detections = []
  all_ground_truths = []

  with torch.no_grad():
    for images, targets in tqdm(data_loader,
                                desc="Evaluating Object Detection"):
      images = list(img.to(device) for img in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      # 추론
      outputs = model(images)

      # 결과 추출
      for i, (output, target) in enumerate(zip(outputs, targets)):
        # 예측 박스, 점수, 라벨
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()

        # 그라운드 트루스 박스, 라벨
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        all_detections.append({
          'boxes': boxes,
          'scores': scores,
          'labels': labels
        })

        all_ground_truths.append({
          'boxes': gt_boxes,
          'labels': gt_labels
        })

  # mAP 계산 (개선 필요)
  # 실제 구현에서는 COCO API 등을 사용하여 정확한 mAP 계산 권장

  return {
    'mAP': 0.0,  # 임시 값
    'precision': 0.0,
    'recall': 0.0
  }