# 🛡️ ExamSentinelX AI — Online Exam Cheating Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-yellow?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-blueviolet?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-3-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**An AI-powered real-time proctoring system that detects cheating behaviors during online exams using computer vision and deep learning.**

*Based on IEEE ACCESS 2024 — "Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams" (DOI: 10.1109/ACCESS.2024.3359689)*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [AI Detection Pipeline](#ai-detection-pipeline)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Dataset Information](#dataset-information)
- [Results & Accuracy](#results--accuracy)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

ExamSentinelX AI is a full-stack web application that monitors students during online exams using AI. It detects five categories of suspicious behavior (inspired by the S_OCA dataset from the IEEE paper):

| Category | Description | Detection Method |
|----------|-------------|-----------------|
| `face_absent` | Student leaves camera frame | MediaPipe Face Detection |
| `multiple_persons` | More than one person visible | MediaPipe + YOLOv8 |
| `looking_away` | Head turned beyond threshold | MediaPipe FaceMesh + solvePnP |
| `phone_detected` | Mobile phone visible | YOLOv8 (COCO Class 67) |
| `suspicious_object` | Book/tablet/notes visible | YOLOv8 (COCO Classes) |

---

## ✨ Features

### Student Interface
- 🔐 Secure registration and login
- 📚 Browse and take available exams
- ⏱️ Real-time countdown timer
- 💾 Auto-save answers
- 📷 Webcam monitoring with live AI annotations
- ⚠️ Instant violation alerts
- 📊 Personal exam history and scores

### Admin Dashboard
- 📊 Overview statistics with charts
- 👥 Student management
- 🔍 Session detail view
- 📄 PDF report generation per session
- 🚩 Flagged session monitoring
- ➕ Create new exams

### AI Module
- **Face Detection**: MediaPipe BlazeFace model
- **Head Pose Estimation**: MediaPipe FaceMesh + OpenCV solvePnP
- **Object Detection**: YOLOv8n (Ultralytics) pretrained on COCO
- **Real-time processing**: Every 2 seconds
- **Evidence capture**: Timestamped screenshots saved for each violation

---

## 🔧 Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Backend | Python 3.10 + Flask 3.0 | Lightweight, perfect for ML integration |
| Database | SQLite + SQLAlchemy | Zero-config, file-based, ideal for MVP |
| AI — Object Detection | YOLOv8 (Ultralytics) | Better than paper's YOLOv5, faster, higher mAP |
| AI — Face/Pose | MediaPipe (Google) | CPU-optimized, production-ready |
| AI — Frame Processing | OpenCV 4.10 | Industry standard for video |
| Frontend | HTML + CSS + Vanilla JS | No framework overhead, full control |
| Real-time | Flask-SocketIO + Eventlet | WebSocket support for live monitoring |
| Reports | fpdf2 | Pure-Python PDF generation |

---

## 📁 Project Structure

```
ExamSentinelXAI/
├── run.py                    # ← Start here
├── requirements.txt
├── .env.example              # Copy to .env and configure
├── .gitignore
│
├── ai/                       # 🤖 AI Detection Modules
│   ├── detector.py           # Main orchestrator
│   ├── face_detector.py      # MediaPipe face detection
│   ├── head_pose_estimator.py # Head pose via solvePnP
│   ├── object_detector.py    # YOLOv8 object detection
│   └── models/               # Downloaded model weights
│
├── backend/                  # 🔧 Flask Application
│   ├── app.py                # Application factory
│   ├── config.py             # Configuration management
│   ├── extensions.py         # Flask extensions
│   ├── auth/routes.py        # Login, register, logout
│   ├── exam/routes.py        # Exam session management
│   ├── monitoring/routes.py  # AI analysis API
│   ├── monitoring/socket_events.py  # WebSocket handlers
│   ├── admin/routes.py       # Admin dashboard
│   └── reports/generator.py # PDF generation
│
├── database/
│   ├── models.py             # SQLAlchemy ORM models
│   └── seed.py               # Sample data seeder
│
├── frontend/
│   ├── static/css/main.css   # Design system
│   ├── static/js/
│   │   ├── webcam.js         # Camera access & capture
│   │   ├── monitor.js        # Real-time AI monitoring
│   │   └── exam.js           # Exam timer & logic
│   └── templates/            # Jinja2 HTML templates
│
├── captures/evidence/        # Violation screenshots
├── reports/generated/        # PDF reports
└── tests/                    # Unit tests
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Webcam
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ExamSentinelXAI.git
cd ExamSentinelXAI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your settings (optional for development)

# 5. Run the application
python run.py
```

### 6. Open in Browser
```
http://localhost:5000
```

### Default Accounts (auto-created on first run)

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `Admin@123456` |
| Student | `john_doe` | `Student@123` |
| Student | `jane_smith` | `Student@123` |

---

## 🤖 AI Detection Pipeline

```
Webcam Frame (640×480)
        │
        ▼
  [OpenCV Preprocessing]
  Gaussian blur, histogram equalization, resize
        │
   ┌────┴─────────────────────────────────┐
   │                                      │
   ▼                                      ▼
[MediaPipe Face Detection]        [YOLOv8n Inference]
   - Count faces                    - Cell phone (class 67)
   - face_absent if 0               - Person (class 0)
   - multiple_persons if >1         - Book, laptop, etc.
   │
   ▼
[MediaPipe FaceMesh]
   - 468 facial landmarks
   - OpenCV solvePnP → rotation matrix
   - Extract Yaw/Pitch/Roll angles
   - looking_away if |Yaw| > 30° or |Pitch| > 20°
        │
        ▼
  [Violation Priority Logic]
  phone > multiple_persons > suspicious_object > face_absent > looking_away
        │
        ▼
  [If violation: Save to DB + Screenshot]
        │
        ▼
  [Return JSON result to browser]
```

---

## 📡 API Documentation

### POST `/monitor/frame`
Send a webcam frame for AI analysis.

**Request:**
```json
{
  "session_id": 42,
  "frame": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response (violation):**
```json
{
  "has_violation": true,
  "violation_type": "phone_detected",
  "confidence": 0.873,
  "face_count": 1,
  "total_violations": 3,
  "risk_score": 4.5,
  "alert_message": "Mobile phone detected! Put it away immediately.",
  "annotated_frame": "data:image/jpeg;base64,...",
  "session_status": "active"
}
```

### POST `/exam/answer`
Auto-save student answer.

**Request:**
```json
{
  "session_id": 42,
  "question_id": 7,
  "selected_answer": "B"
}
```

### POST `/exam/submit/<session_id>`
Submit the exam and calculate score.

**Response:**
```json
{
  "status": "submitted",
  "score": 35,
  "total": 50,
  "session_status": "completed",
  "redirect": "/exam/dashboard"
}
```

---

## 📊 Results & Accuracy

Based on the IEEE paper (S_OCA Dataset):

| Model | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| **YOLOv5** (paper) | 95.54% | 93.16% | **95.40%** |
| Inception_ResNet_v2 | ~87% | ~85% | — |
| DenseNet121 | ~84% | ~82% | — |
| Inception-V3 | ~86% | ~84% | — |
| Custom CNN | ~80% | ~78% | — |

**Our Implementation:** Uses YOLOv8 (successor to YOLOv5) which achieves:
- YOLOv8n: mAP@0.5 = 37.3 on COCO (vs YOLOv5n's 28.0)
- ~2× faster inference

---

## 🔮 Future Improvements

- [ ] Train custom YOLOv8 model on S_OCA dataset (currently using COCO pretrained)
- [ ] Add eye gaze tracking
- [ ] Audio analysis (whisper/talking detection)
- [ ] Screen activity monitoring (tab switching)
- [ ] Student identity verification at session start
- [ ] Multi-camera support
- [ ] Docker deployment configuration
- [ ] REST API for LMS integration (Moodle, Canvas)

---

## 📚 References

1. M. Ramzan et al., "Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams," *IEEE Access*, vol. 12, 2024. DOI: 10.1109/ACCESS.2024.3359689
2. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
3. MediaPipe: https://developers.google.com/mediapipe
4. OpenCV: https://opencv.org

---

## 📄 License

This project is for educational purposes. MIT License.

---

<div align="center">
  <strong>Built with ❤️ for internship submission | ExamSentinelX AI v1.0</strong>
</div>
