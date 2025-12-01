from ultralytics import YOLO
import torch

# 1. Load via Ultralytics first (handles architecture definitions automatically)
yolo_wrapper = YOLO("yolov11n.pt")

# 2. Extract the underlying PyTorch model
pytorch_model = yolo_wrapper.model



