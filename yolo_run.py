from ultralytics import YOLO
import os

image_path = "coffe.jpeg"

# Check if image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"{image_path} does not exist!")

# Detection
det_model = YOLO("yolov8n.pt")
det_model.predict(image_path, conf=0.5, save=True)
print("Detection results saved in 'runs/detect/predict/'")

# Segmentation
seg_model = YOLO("yolov8n-seg.pt")
seg_model.predict(image_path, conf=0.5, save=True)
print("Segmentation results saved in 'runs/segment/predict/'")
