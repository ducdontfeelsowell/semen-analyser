from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\admin\PycharmProjects\SemenAnalyser\train_script\wakuwaku_1.pt")  # Replace with your model path

# Load an image
image_path = r"C:\Users\admin\PycharmProjects\SemenAnalyser\dataset\train\images\Abnormal_Sperm-1-_bmp.rf.28a735dd5989c57e624027a5ddf31792.jpg"  # Replace with the correct path
image = cv2.imread(image_path)

# Run predictions with a confidence threshold
results = model.predict(source=image, save=False, device="cuda", conf=0.7)

# Convert to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a Matplotlib figure
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)

# Overlay predictions
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
    conf = box.conf[0].cpu().numpy()  # Confidence score
    cls = int(box.cls[0].cpu().numpy())  # Class ID
    label = f"{model.names[cls]} {conf:.2f}"
    class_name = model.names[cls]  # Class label name (e.g., "filled" or "unfilled")

    # Draw bounding boxes and labels only for high-confidence predictions
    if conf > 0.7 : # class_name == "filled":  # Adjust threshold as needed
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          edgecolor="red", linewidth=2, fill=False))
        plt.text(x1, y1 - 5, label, color="red", fontsize=8, weight="bold")

# Remove axes and show the image
plt.axis("off")
plt.show()
