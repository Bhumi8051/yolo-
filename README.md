# Self-Driving Cars Object Detection with YOLO

A computer vision project for detecting objects in self-driving car scenarios using YOLOv8 (You Only Look Once) deep learning model. This project focuses on detecting and classifying objects commonly found in autonomous driving environments.

## 🚗 Overview

This project implements real-time object detection for self-driving car applications using the YOLOv8 model from Ultralytics. The model can detect and classify 5 different object types that are crucial for autonomous vehicle navigation.

## ✨ Features

- **Object Detection**: Detects multiple objects in a single image
- **5 Class Detection**: Identifies cars, trucks, pedestrians, bicycles, and traffic lights
- **Bounding Box Visualization**: Displays detected objects with confidence scores
- **Real-time Inference**: Fast object detection using YOLOv8 medium model
- **Dataset Analysis**: Tools for exploring and visualizing labeled datasets

## 🎯 Detected Classes

The model detects the following 5 classes:

1. **Car** (class_id: 1)
2. **Truck** (class_id: 2)
3. **Person/Pedestrian** (class_id: 3)
4. **Bicycle** (class_id: 4)
5. **Traffic Light** (class_id: 5)

## 📋 Requirements

### Python Packages

```bash
pip install ultralytics
pip install numpy
pip install pandas
pip install opencv-python
pip install matplotlib
pip install scikit-learn
pip install pillow
```

### Model

- **YOLOv8 Medium** (`yolov8m.pt`) - Automatically downloaded on first use

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/Bhumi8051/yolo-.git
cd yolo-
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook Self-DrivingCars.ipynb
```

## 📁 Project Structure

```
yolo/
├── Self-DrivingCars.ipynb          # Main Jupyter notebook
├── Object detection/
│   ├── images/                      # Training images (excluded from git)
│   ├── labels_train.csv            # Training labels
│   ├── labels_trainval.csv         # Train-validation labels
│   └── labels_val.csv              # Validation labels
├── class_1_car.png                 # Example: Car class
├── class_2_truck.png               # Example: Truck class
├── class_3_person.png              # Example: Person class
├── class_4_bicycle.png             # Example: Bicycle class
├── class_5_traffic light.png       # Example: Traffic light class
├── image_*.jpg.png                 # Sample detection images
└── README.md                       # This file
```

## 💻 Usage

### Basic Object Detection

```python
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")

# Run inference on an image
results = model.predict(
    source="path/to/image.jpg",
    save=True,
    conf=0.2,    # Confidence threshold
    iou=0.5      # IoU threshold for NMS
)

# Access detection results
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        print(f"Object: {class_id}, Confidence: {conf:.2f}, Coordinates: {cords}")
```

### Visualizing Results

```python
# Plot results with bounding boxes
plot = results[0].plot()
import cv2
plot_rgb = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot_rgb))
```

## 📊 Dataset

The project uses a labeled dataset with:
- Training images with bounding box annotations
- CSV files containing labels with coordinates (xmin, xmax, ymin, ymax) and class IDs
- Multiple object instances per image support

### Label Format

The CSV files contain the following columns:
- `frame`: Image filename
- `xmin`, `xmax`, `ymin`, `ymax`: Bounding box coordinates
- `class_id`: Object class identifier (1-5)

## 🔍 Example Results

The model successfully detects:
- Multiple cars in traffic scenes
- Trucks and commercial vehicles
- Pedestrians crossing roads
- Bicycles and cyclists
- Traffic lights at intersections

Detection results include:
- Bounding box coordinates
- Class predictions
- Confidence scores

## 🛠️ Technologies Used

- **YOLOv8**: State-of-the-art object detection model
- **Ultralytics**: Python package for YOLO implementation
- **OpenCV**: Image processing and visualization
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computations

## 📝 Notes

- Large image datasets are excluded from the repository via `.gitignore` to keep the repository size manageable
- Model weights (`yolov8m.pt`) are automatically downloaded on first use
- The project is designed for educational and research purposes

## 👤 Author

**Bhumi Jain**

- GitHub: [@Bhumi8051](https://github.com/Bhumi8051)

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- The open-source computer vision community

---

⭐ If you find this project helpful, please consider giving it a star!

