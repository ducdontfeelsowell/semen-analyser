from ultralytics import YOLO

if __name__ == '__main__':

    # Load YOLOv8 model
    model = YOLO("wakuwaku_14_augmented.pt")

    # Train the model
    model.train(
        data=r"C:\Users\admin\PycharmProjects\SemenAnalyser\dataset\data.yaml",
        device="cuda",
        epochs=10,
        batch=4,
        imgsz=1024,
        weight_decay=0.001,  # Weight decay
        lr0=0.002,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        optimizer='AdamW',  # Explicitly specify the optimizer

        hsv_h=0.015,  # Hue saturation value (similar to your HueSaturationValue)
        hsv_s=0.5,  # Saturation shift
        hsv_v=0.4,  # Value shift / brightness -|+60%

        mosaic=1.0,
        mixup=0.2,

        # Brightness/Contrast adjustments
        #brightness=0.2,  # Brightness limit
        #contrast=0.5,  # Contrast limit

        # Scaling (random resize)
        scale=0.8,  # Similar to RandomScale in Albumentations

        # Flip Augmentations
        flipud=0.5, # Vertical flip
        fliplr=0.5  # Horizontal flip

        #degrees=15

        # Gaussian noise (not directly available, but can be achieved with other noise augmentation)
        #noise=0.3

        # CLAHE for local contrast enhancement (not directly available)
        #clahe=0.7,

        # Sharpening (YOLOv8 does not directly support sharpening; you can use external augmentations)
        #sharpen=0.7

        # Other augmentations can be added similarly
    )
    model.save('wakuwaku_14_augmented.pt')
