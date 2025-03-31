# blind-savers/apps/vision/src/inference/inference.py
import sys
import os

sys.path.append(
  os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ..models.model import VisionModel
from ..utils.preprocessing import get_transform, preprocess_for_inference
from PIL import Image
import torch
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {"num_classes": 3, "model_path": "checkpoints/vision_model.pth"}
class_names = ["braille_block", "cup", "chair"]
transform = get_transform()
model = None


def load_model(model_path, num_classes, device):
  global model
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


@app.route("/predict", methods=["POST"])
def predict():
  if "image" not in request.files:
    return jsonify({"error": "No image provided"}), 400

  file = request.files["image"]
  image = Image.open(file.stream).convert("RGB")
  prediction = infer_image(model, image, transform, device, class_names)
  return jsonify({"prediction": prediction})


if __name__ == "__main__":
  load_model(config["model_path"], config["num_classes"], device)
  app.run(host="0.0.0.0", port=5000)

  # React에서 부를때
  # const sendImageToServer = async (imageUri) => {
  #   const formData = new FormData();
  #   formData.append("image", {
  #     uri: imageUri,
  #     type: "image/jpeg",
  #     name: "test.jpg",
  #   });
  #
  #   const response = await fetch("http://<server-ip>:5000/predict", {
  #     method: "POST",
  #     body: formData,
  #   });
  #   const result = await response.json();
  #   console.log(result.prediction); // "braille_block" 등
  # };