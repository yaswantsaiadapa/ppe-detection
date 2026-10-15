# PPE Compliance Detection System using YOLOv8

## Overview

This project implements a Personal Protective Equipment (PPE) Compliance Detection System using YOLOv8 for object detection. The objective is to automatically detect PPE equipment and safety violations in workplace surveillance images.

The model is capable of detecting the following classes:

### PPE Classes

* Boots
* Gloves
* Goggles
* Helmet
* Vest

### Violation Classes

* No-Boots
* No-Gloves
* No-Goggles
* No-Helmet
* No-Vest

The system was developed as part of the DECTIFY Machine Learning Intern Assessment.

---

## Dataset

Dataset Source: Roboflow PPE Detection Dataset

Dataset Statistics:

| Split      | Images |
| ---------- | ------ |
| Train      | 3597   |
| Validation | 1026   |
| Test       | 517    |
| Total      | 5140   |

Total Classes: 10

The dataset is provided in YOLO object detection format with bounding box annotations.

---

## Model Selection

YOLOv8n was selected because:

* Lightweight architecture
* Fast training and inference
* Excellent transfer learning support
* Suitable for Tesla T4 GPU
* Widely used in real-world computer vision systems

Transfer learning was performed using pretrained COCO weights.

---

## Training Configuration

| Parameter  | Value                 |
| ---------- | --------------------- |
| Model      | YOLOv8n               |
| Epochs     | 30                    |
| Batch Size | 16                    |
| Image Size | 640                   |
| Optimizer  | AdamW (Auto Selected) |
| GPU        | Tesla T4              |
| Framework  | Ultralytics YOLOv8    |

### Augmentations

The following augmentations were automatically applied during training:

* Mosaic Augmentation
* Horizontal Flip
* HSV Color Augmentation
* Blur
* Median Blur
* CLAHE

---

## Results

### Final Metrics

| Metric       | Value |
| ------------ | ----- |
| Precision    | 0.477 |
| Recall       | 0.615 |
| mAP@0.5      | 0.538 |
| mAP@0.5:0.95 | 0.328 |

---

## Class-wise Observations

Strong Performance:

| Class  | mAP@0.5 |
| ------ | ------- |
| Helmet | 0.966   |
| Vest   | 0.961   |
| Boots  | 0.930   |

Weak Performance:

| Class     | mAP@0.5 |
| --------- | ------- |
| No-Boots  | 0.101   |
| No-Gloves | 0.099   |
| No-Helmet | 0.283   |

---

## Root Cause Analysis

The primary challenge observed was severe class imbalance.

Examples:

| Class     | Instances |
| --------- | --------- |
| Helmet    | 1187      |
| Vest      | 1154      |
| Boots     | 931       |
| No-Boots  | 36        |
| No-Gloves | 51        |

The model learned PPE presence classes effectively due to abundant training samples, while PPE violation classes suffered from insufficient representation.

---

## Repository Structure

```text
dectify-ppe-detection/
│
├── PPE_Detection.ipynb
├── README.md
├── requirements.txt
├── Evaluation_Report.md
│
├── weights/
│   └── best.pt
│
└── outputs/
    ├── results.png
    ├── PR_curve.png
    ├── confusion_matrix.png
    ├── prediction1.jpg
    ├── prediction2.jpg
    ├── prediction3.jpg
    ├── prediction4.jpg
    └── prediction5.jpg
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Inference

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    source="image.jpg",
    save=True,
    conf=0.25
)
```

---

## Future Improvements

* Address class imbalance through data collection
* Experiment with YOLOv8s and YOLOv8m
* Hyperparameter tuning
* Real-time CCTV deployment
* Edge-device optimization
* ONNX export and TensorRT acceleration
