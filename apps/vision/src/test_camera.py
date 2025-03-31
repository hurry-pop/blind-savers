# 개발 단계에서 웹캠으로 실시간 테스트 할수있ㄱ께
# blind-savers/apps/vision/src/test_camera.py
# blind-savers/apps/vision/src/test_camera.py
import sys
import os

import torch

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # src/ 경로 추가

from models.model import VisionModel
from utils.preprocessing import get_transform, preprocess_for_inference
from utils.model_utils import load_model
from PIL import Image


def test_file_input(model_path, num_classes, class_names, image_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  transform = get_transform(train=False)
  model = load_model(VisionModel(num_classes), model_path, device)

  # 이미지 파일 로드 및 추론
  image = Image.open(image_path).convert("RGB")
  prediction = infer_image(model, image, transform, device, class_names)
  print(f"Predicted class: {prediction}")


def infer_image(model, image, transform, device, class_names):
  image_tensor = preprocess_for_inference(image, transform).to(device)
  with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
  return class_names[predicted.item()]


if __name__ == "__main__":
  config = {"model_path": "checkpoints/vision_model.pth", "num_classes": 3}
  class_names = ["braille_block", "cup", "chair"]
  image_path = "test_image.jpg"  # 테스트용 이미지 경로
  test_file_input(config["model_path"], config["num_classes"], class_names,
                  image_path)