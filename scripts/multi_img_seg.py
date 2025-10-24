# multi_img_seg.py
# Task 2: Multiple Image Object Detection & Segmentation
# Author: Ruqsar Firdous

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Paths configuration
# ----------------------------
BASE_DIR = Path("../")                       # Task2_MultiImage main folder
IMAGE_DIR = BASE_DIR / "images"              # Folder with input images
DETECT_OUT = BASE_DIR / "outputs/detect"     # Folder to save detection results locally
SEGMENT_OUT = BASE_DIR / "outputs/segment"   # Folder to save segmentation results locally
YOLO_DET = BASE_DIR / "yolov8x.pt"
YOLO_SEG = BASE_DIR / "yolov8x-seg.pt"

# Create output folders if they don't exist
DETECT_OUT.mkdir(parents=True, exist_ok=True)
SEGMENT_OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load YOLO models
# ----------------------------
det_model = YOLO(str(YOLO_DET))       # Object detection model
seg_model = YOLO(str(YOLO_SEG))       # Segmentation model

# ----------------------------
# Object Detection on multiple images
# ----------------------------
print("\n Running Object Detection on multiple images...\n")
for img_path in tqdm(sorted(IMAGE_DIR.glob("*.*")), desc="Detection Progress"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # Predict and save to runs/detect automatically
    results = det_model.predict(
        source=str(img_path),
        conf=0.5,
        imgsz=1024,
        save=True,                                  # Enable saving
        project=str(BASE_DIR / "runs/detect"),      # Folder for YOLO runs
        name="exp",                                 # Subfolder for this run
        exist_ok=True                               # Overwrite if folder exists
    )

    # Also save locally in outputs/detect
    for r in results:
        annotated = r.plot()
        Image.fromarray(annotated).save(DETECT_OUT / img_path.name)

print(f"\n Detection complete! Check 'runs/detect/exp/' and '{DETECT_OUT}' for results.\n")

# ----------------------------
# Segmentation on multiple images
# ----------------------------
print("\n🎨 Running Image Segmentation on multiple images...\n")
for img_path in tqdm(sorted(IMAGE_DIR.glob("*.*")), desc="Segmentation Progress"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # Predict and save to runs/segment automatically
    results = seg_model.predict(
        source=str(img_path),
        conf=0.5,
        imgsz=1024,
        save=True,                                  # Enable saving
        project=str(BASE_DIR / "runs/segment"),     # Folder for YOLO runs
        name="exp",                                 # Subfolder for this run
        exist_ok=True                               # Overwrite if folder exists
    )

    # Also save locally in outputs/segment
    for r in results:
        annotated = r.plot()
        Image.fromarray(annotated).save(SEGMENT_OUT / img_path.name)

print(f"\n Segmentation complete! Check 'runs/segment/exp/' and '{SEGMENT_OUT}' for results.\n")
print("✅ All images processed successfully! You can now check runs/ and outputs/ folders.")
