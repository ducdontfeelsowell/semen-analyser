# export_model.py

from ultralytics import YOLO

def export_model():
    model = YOLO("sperm_detection/yolov8s/weights/best.pt")

    model.export(format="onnx")  # Options: torchscript, tflite, coreml, etc.

if __name__ == "__main__":
    export_model()
