import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# from ultralytics import YOLO
# model = YOLO("yolov8s.pt")
# print(model.device)  # Should say "cuda:0" if it's using GPU
