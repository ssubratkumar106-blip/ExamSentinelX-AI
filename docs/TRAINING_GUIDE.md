# ExamSentinelX AI — Dataset & Model Training Guide

## Overview

ExamSentinelX AI uses a **3-layer detection pipeline** based on the IEEE paper:
*"Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams"*

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 1 | MediaPipe (FaceMesh) | Face presence, count, head pose |
| 2 | YOLOv8n (fine-tuned) | Object detection (phone, book, person) |
| 3 | InceptionV3 + DenseNet121 | Frame-level behavior classification |

---

## Step 1: Download the Dataset

### Option A — Roboflow SDK (Recommended)

```bash
# 1. Get your FREE API key at: https://app.roboflow.com/settings/api

# 2. Set environment variable
set ROBOFLOW_API_KEY=your_api_key_here

# 3. Run downloader
.\venv\Scripts\python.exe ai/download_dataset.py
```

### Option B — Manual Download

1. Go to: **https://universe.roboflow.com/cheating-online-exam/cheating-online-exam**
2. Click **"Download Dataset"**
3. Select format: **YOLOv8** (for object detection training)
4. Extract ZIP to: `ExamSentinelXAI/ai/datasets/cheating_exam/`

Expected folder structure after extraction:
```
ai/datasets/cheating_exam/
├── train/
│   ├── images/    ← .jpg files
│   └── labels/    ← .txt YOLO annotations
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml      ← class definitions
```

### Other Recommended Datasets (upload later as user mentioned)

| Dataset | URL | Format | Classes |
|---------|-----|--------|---------|
| Cheating Detection Refined | https://universe.roboflow.com/fyp-terry-lee/cheating-detection-refined | YOLOv8 | cheating, normal |
| ExamCheating (Kaggle) | https://www.kaggle.com/datasets/ardutraagiginting/examcheating-dataset | Images | 5 behaviors |
| OEP Dataset (Kaggle) | https://www.kaggle.com/datasets/raajanwankhade/oep-dataset | Video/Images | cheating, normal |
| MSU Proctoring | https://www.kaggle.com/datasets/elvinagammedova/msu-online-exam-proctoring-dataset | Video | behavioral |

---

## Step 2: Train YOLOv8 (Object Detection)

```bash
# Basic training (CPU, 50 epochs)
.\venv\Scripts\python.exe ai/train_model.py --mode yolo --epochs 50 --batch 8 --device cpu

# GPU training (much faster)
.\venv\Scripts\python.exe ai/train_model.py --mode yolo --epochs 100 --batch 16 --device 0

# After training, the best model is saved to:
# ai/models/yolov8_cheating_exam.pt
# The system auto-loads it on next startup!
```

**Expected Results after fine-tuning:**
- mAP50: ~0.75–0.85 (vs 0.45 baseline COCO)
- Phone detection: ~85% accuracy
- Book detection: ~70% accuracy

---

## Step 3: Train CNN Classifiers (Pre-trained Transfer Learning)

```bash
# Train InceptionV3 (best accuracy, IEEE paper: 94.2%)
.\venv\Scripts\python.exe ai/train_model.py --mode cnn --model InceptionV3 --epochs 30

# Train DenseNet121 (fast, IEEE paper: 92.8%)
.\venv\Scripts\python.exe ai/train_model.py --mode cnn --model DenseNet121 --epochs 30

# Train both (ensemble = highest accuracy)
.\venv\Scripts\python.exe ai/train_model.py --mode all --epochs 50
```

**Note:** Requires TensorFlow:
```bash
.\venv\Scripts\pip.exe install tensorflow
```

---

## Step 4: Evaluate Trained Models

```bash
.\venv\Scripts\python.exe ai/train_model.py --mode evaluate
```

---

## Step 5: Add More Datasets

When you upload more datasets, merge them before training:

```bash
# If you have a second dataset in ai/datasets/dataset2/
# Merge into main dataset folder, then retrain:
.\venv\Scripts\python.exe ai/train_model.py --mode yolo --epochs 100
```

---

## Model Auto-Loading Priority

The system automatically selects the best available model:

```
1. ai/models/yolov8_cheating_exam.pt   ← Fine-tuned (best)
2. yolov8n.pt                           ← COCO pretrained (fallback)

For CNN:
1. ai/models/InceptionV3_exam.h5       ← Fine-tuned (used if TF available)
2. ai/models/DenseNet121_exam.h5       ← Fine-tuned (ensemble)
3. No CNN                               ← Graceful fallback
```

---

## Hardware Requirements

| Mode | Min RAM | Recommended | GPU |
|------|---------|------------|-----|
| Run (inference only) | 4 GB | 8 GB | Not required |
| Train YOLOv8 (CPU) | 8 GB | 16 GB | Optional |
| Train YOLOv8 (GPU) | 6 GB VRAM | 8 GB VRAM | CUDA 11.8+ |
| Train CNN (CPU) | 8 GB | 16 GB | Not recommended |
| Train CNN (GPU) | 4 GB VRAM | 8 GB VRAM | CUDA 11.8+ |

---

## Training Tips

1. **Small dataset?** Use more augmentation:
   ```python
   # In train_model.py, increase augmentation parameters
   augment=True, mosaic=1.0, mixup=0.2
   ```

2. **Low accuracy?** Add more data from Kaggle/Roboflow:
   ```bash
   # Download more datasets, merge images/labels, retrain
   ```

3. **Slow training?** Reduce batch size or use smaller model:
   ```bash
   --batch 4 --model MobileNetV2
   ```

4. **Overfitting?** Increase dropout or reduce epochs:
   ```bash
   --epochs 20
   ```
