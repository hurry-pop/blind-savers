import cv2
import numpy as np
from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image):
    """기본 이미지 전처리 함수"""
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 