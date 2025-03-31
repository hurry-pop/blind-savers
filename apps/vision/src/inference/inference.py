# React-Native에서 실시간 이미지 처리를 위해 모델 추론 로직
# 카메라 입력을 시뮬레이션하고 결과를 테스트할 수 있움
# blind-savers/apps/vision/src/inference/inference.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # src/ 경로 추가

from ..models.model import VisionModel
from ..utils.preprocessing import get_transform, preprocess_for_inference
from PIL import Image
import torch

def load_model(model_path, num_classes, device):
    model = VisionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_image(model, image, transform, device, class_names):
    image_tensor = preprocess_for_inference(image, transform).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "num_classes": 3,  # config.yaml에서 가져오거나 하드코딩
        "model_path": "checkpoints/vision_model.pth"
    }
    class_names = ["braille_block", "cup", "chair"]  # 클래스 이름 목록

    # 모델 로드
    transform = get_transform()
    model = load_model(config["model_path"], config["num_classes"], device)

    # 테스트 이미지 로드 및 추론
    test_image = Image.open("test_image.jpg").convert("RGB")  # Pillow로 이미지 로드
    prediction = infer_image(model, test_image, transform, device, class_names)
    print(f"Predicted class: {prediction}")