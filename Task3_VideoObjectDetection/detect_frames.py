from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt for better accuracy if you want

# Input and output folder paths
input_folder = "frames"
output_folder = "processed_frames"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# Process each frame
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".jpg"):
        frame_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # Run YOLO detection
        results = model(frame_path)
        
        # Save the result image with bounding boxes
        results[0].save(filename=output_path)

print("✅ All frames processed and saved in 'processed_frames' folder!")
