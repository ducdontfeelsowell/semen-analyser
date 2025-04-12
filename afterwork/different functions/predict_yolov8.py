# predict_yolov8.py

from ultralytics import YOLO
import cv2

def predict(image_path):
    model = YOLO(r"C:\Users\admin\PycharmProjects\SemenAnalyser\train_script\wakuwaku_1.pt")

    results = model(image_path)

    # Display results
    results[0].show()

    # Optional: Save results
    results[0].save(filename="predicted.jpg")

if __name__ == "__main__":
    predict(r"C:\Users\admin\PycharmProjects\SemenAnalyser\dataset\test\images\Normal_Sperm-14-_bmp.rf.6e633fea5493939333f9ec79a34d82ce.jpg")
