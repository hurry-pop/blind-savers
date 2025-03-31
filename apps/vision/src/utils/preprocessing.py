# 전처리
# blind-savers/apps/vision/src/utils/preprocessing.py
from PIL import Image
import numpy as np
from torchvision import transforms

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def preprocess_image(image):
    if isinstance(image, str):  # 파일 경로일 경우
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):  # numpy 배열일 경우
        image = Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, Image.Image):  # 이미 PIL 이미지일 경우
        image = image.convert("RGB")
    else:
        raise ValueError("이미지 형식이 지원되지 않습니다.")
    return image

def preprocess_for_inference(image, transform):
    image = preprocess_image(image)
    image = transform(image)
    return image.unsqueeze(0)