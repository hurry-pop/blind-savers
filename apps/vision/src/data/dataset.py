# 사물 인식과 점자블록 인식을 위해 라벨을 추가하고, 디렉토리 구조를 활용해 클래스를 자동으로 추출하도록 수정
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class VisionDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform
    self.classes = sorted(
      os.listdir(data_dir))  # 클래스 목록 (예: "braille_block", "cup")
    self.class_to_idx = {cls_name: idx for idx, cls_name in
                         enumerate(self.classes)}
    self.image_files = []

    # 클래스별 이미지 파일 수집
    for cls_name in self.classes:
      cls_dir = os.path.join(data_dir, cls_name)
      for img_file in os.listdir(cls_dir):
        self.image_files.append((os.path.join(cls_dir, img_file), cls_name))

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_path, label = self.image_files[idx]
    image = Image.open(img_path).convert('RGB')

    if self.transform:
      image = self.transform(image)

    label_idx = self.class_to_idx[label]
    return image, label_idx

# 사용 예시 데이터 구조:
# data/
# ├── braille_block/
# │   ├── img1.jpg
# │   └── img2.jpg
# ├── cup/
# │   ├── img3.jpg
# └── chair/
#     └── img4.jpg