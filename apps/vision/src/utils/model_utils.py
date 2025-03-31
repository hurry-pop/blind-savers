# 모델을 저장하고 로드하는 과정을 표준화하기 위해 유틸리티 함수
import torch
import os

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# 사용예시
# from utils.model_utils import save_model, load_model
# save_model(model, "checkpoints/vision_model.pth")
# model = load_model(VisionModel(num_classes), "checkpoints/vision_model.pth", device)