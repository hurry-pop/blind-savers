import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path


class VisionDataset(Dataset):
  """
    사물 인식을 위한 이미지 데이터셋 클래스
    """

  def __init__(self, data_dir, transform=None, split='train', label_file=None):
    """
        매개변수:
            data_dir (str): 이미지 파일이 저장된 디렉토리 경로
            transform (callable, optional): 이미지에 적용할 변환
            split (str): 'train', 'val', 'test' 중 하나
            label_file (str, optional): 라벨 정보가 담긴 JSON 파일 경로
        """
    self.data_dir = Path(data_dir)
    self.transform = transform
    self.split = split

    # 이미지 파일 목록 수집
    self.image_files = [f for f in os.listdir(data_dir) if
                        f.endswith(('.jpg', '.jpeg', '.png'))]

    # 라벨 파일이 있으면 로드
    self.labels = {}
    if label_file and os.path.exists(label_file):
      with open(label_file, 'r') as f:
        self.labels = json.load(f)
    else:
      # 라벨 파일이 없으면 디렉토리 구조에서 추론
      for img_file in self.image_files:
        # 파일명에서 클래스 추출 (예: tactile_paving_001.jpg -> tactile_paving)
        class_name = img_file.split('_')[0]
        self.labels[img_file] = class_name

    # 클래스 이름을 인덱스로 매핑
    self.class_to_idx = {}
    unique_classes = sorted(set(self.labels.values()))
    for i, cls in enumerate(unique_classes):
      self.class_to_idx[cls] = i

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    """
        특정 인덱스의 이미지와 라벨을 반환
        """
    img_file = self.image_files[idx]
    img_path = os.path.join(self.data_dir, img_file)

    # 이미지 로드
    try:
      image = Image.open(img_path).convert('RGB')
    except Exception as e:
      print(f"Error loading image {img_path}: {e}")
      # 에러 발생 시 더미 이미지 반환
      image = Image.new('RGB', (224, 224), color='black')

    # 변환 적용
    if self.transform:
      image = self.transform(image)

    # 라벨 정보 가져오기
    label_name = self.labels.get(img_file, "unknown")
    label_idx = self.class_to_idx.get(label_name, -1)

    return image, torch.tensor(label_idx, dtype=torch.long)

  def get_class_distribution(self):
    """
        데이터셋의 클래스 분포 반환
        """
    distribution = {}
    for img_file in self.image_files:
      label = self.labels.get(img_file, "unknown")
      if label in distribution:
        distribution[label] += 1
      else:
        distribution[label] = 1
    return distribution


class ObjectDetectionDataset(Dataset):
  """
    객체 감지를 위한 데이터셋 클래스
    """

  def __init__(self, data_dir, annotation_dir, transform=None, split='train'):
    """
        매개변수:
            data_dir (str): 이미지 파일이 저장된 디렉토리 경로
            annotation_dir (str): 어노테이션 파일이 저장된 디렉토리 경로
            transform (callable, optional): 이미지에 적용할 변환
            split (str): 'train', 'val', 'test' 중 하나
        """
    self.data_dir = Path(data_dir)
    self.annotation_dir = Path(annotation_dir)
    self.transform = transform
    self.split = split

    # 이미지 파일 목록 수집
    self.image_files = [f for f in os.listdir(data_dir) if
                        f.endswith(('.jpg', '.jpeg', '.png'))]

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    """
        특정 인덱스의 이미지와 바운딩 박스를 반환
        """
    img_file = self.image_files[idx]
    img_path = os.path.join(self.data_dir, img_file)

    # 이미지 로드
    image = Image.open(img_path).convert('RGB')

    # 어노테이션 로드 (COCO 형식 가정)
    ann_file = img_file.replace('.jpg', '.json').replace('.jpeg',
                                                         '.json').replace(
      '.png', '.json')
    ann_path = os.path.join(self.annotation_dir, ann_file)

    boxes = []
    labels = []

    if os.path.exists(ann_path):
      with open(ann_path, 'r') as f:
        annotations = json.load(f)

      for ann in annotations['annotations']:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2] 형식으로 변환
        labels.append(ann['category_id'])

    # 변환 적용
    if self.transform:
      image = self.transform(image)

    # 텐서로 변환
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    # 대상 정보를 딕셔너리로 구성
    target = {
      'boxes': boxes,
      'labels': labels,
      'image_id': torch.tensor([idx]),
      'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
      'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
    }

    return image, target