from ultralytics import YOLO
import os
import cv2

# Load trained model
model = YOLO(r"C:\Users\admin\PycharmProjects\SemenAnalyser\train_script\wakuwaku_14_augmented.pt")

# Input and output directories
input_dir = r"C:\Users\admin\PycharmProjects\SemenAnalyser\dataset\test\images"
output_dir = "inference_output"
os.makedirs(output_dir, exist_ok=True)

# Get list of image filenames
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Run inference and save results with original filenames
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    result = model(image_path, conf=0.4)[0]

    # Plot and save using original filename
    output_path = os.path.join(output_dir, image_file)
    result.plot(save=True, filename=output_path)
