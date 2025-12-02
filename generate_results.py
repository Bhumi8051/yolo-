"""
Script to generate result images with YOLO predictions
"""
from ultralytics import YOLO
import cv2
import os
from PIL import Image

# Create results directory
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Load the model
print("Loading YOLOv8 model...")
model = YOLO("yolov8m.pt")

# Images used in the notebook for predictions
test_images = [
    "Object detection/images/1478019956680248165.jpg",  # 1 car, 1 truck
    "Object detection/images/1478020211690815798.jpg",  # 4 cars, 3 traffic lights
]

# Generate predictions and save results
for img_path in test_images:
    if os.path.exists(img_path):
        print(f"\nProcessing: {img_path}")
        
        # Run prediction
        results = model.predict(
            source=img_path,
            save=False,
            conf=0.2,
            iou=0.5
        )
        
        # Get the plotted result
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Save the result
        filename = os.path.basename(img_path).replace('.jpg', '_result.jpg')
        output_path = os.path.join(results_dir, filename)
        cv2.imwrite(output_path, result_img)
        
        print(f"Saved result to: {output_path}")
        print(f"Detections: {len(results[0].boxes)} objects")
        
        # Print detection details
        for box in results[0].boxes:
            class_id = results[0].names[box.cls[0].item()]
            conf = round(box.conf[0].item(), 2)
            print(f"  - {class_id}: {conf:.2f} confidence")
    else:
        print(f"Image not found: {img_path}")

print(f"\n✅ All result images saved to '{results_dir}' directory!")

