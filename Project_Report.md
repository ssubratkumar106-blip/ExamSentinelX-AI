# PROJECT REPORT
**Project Title:** ExamSentinelX AI: Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams
**Author:** Subrat Kumar Sahoo
**Date:** July 2026
**Institution:** Xtragrad Internship Program

---

## CHAPTER 1: INTRODUCTION
The rapid expansion of online education, accelerated by global events such as the COVID-19 pandemic, has made online examinations a standard evaluation method. However, the lack of physical invigilation introduces significant challenges in maintaining academic integrity. Online exam environments are highly susceptible to cheating behaviors, including the use of external devices, unauthorized assistance from multiple people, and referring to physical notes. 

The primary objective of the ExamSentinelX AI project is to develop an automated, intelligent framework capable of monitoring examinees and identifying abnormal activities in real time. This system aims to provide a reliable, scalable, and non-intrusive alternative to traditional human proctoring. By leveraging advanced deep learning architectures and computer vision algorithms, ExamSentinelX AI analyzes continuous video streams from a student's webcam to flag potential misconduct, thereby preserving the credibility of online assessments.

## CHAPTER 2: SYSTEM OVERVIEW
ExamSentinelX AI is a comprehensive, real-time proctoring system designed to detect and classify cheating behaviors during online examinations using computer vision and deep learning techniques. The system detects five distinct abnormal behaviors: face absence, presence of multiple persons, looking away, detecting suspicious objects (e.g., books), and mobile phone usage. 

The system operates using a client-server model. A student takes an exam via a web interface, which continuously streams webcam frames to a Python/Flask backend server. The backend runs an optimized AI detection pipeline that evaluates the frame, determines if a violation has occurred using a prioritized logic system, records the incident, and saves an evidence screenshot. An administrative dashboard provides real-time monitoring, session review, and automated PDF reporting of any flagged behaviors.

## CHAPTER 3: LITERATURE SURVEY
The methodology for ExamSentinelX AI is deeply inspired by the S_OCA dataset and approaches outlined in the IEEE ACCESS 2024 paper "Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams" (Ramzan et al.). The literature survey highlighted the need for robust real-time detection systems capable of identifying nuanced cheating behaviors.

Prior research predominantly focused on simple facial recognition or isolated object detection. Ramzan et al. demonstrated the efficacy of utilizing pre-trained Convolutional Neural Networks (CNNs) like YOLOv5, InceptionV3, DenseNet121, and Inception-ResNet-v2, alongside a custom CNN, to classify normal versus abnormal behaviors (such as external devices, head movement, multiple persons, and talking). Building upon this literature, ExamSentinelX AI integrates these foundational concepts, upgrading components (e.g., YOLOv8 for improved speed and mAP) and incorporating lightweight geometric solutions (MediaPipe) to construct a highly efficient, real-time detection pipeline suitable for web deployment.

## CHAPTER 4: TECHNOLOGIES USED
- **Programming Languages:** Python 3.10+, JavaScript (Vanilla), HTML5, CSS3
- **Web Framework:** Flask 3.0, Flask-SocketIO (for real-time WebSocket communication)
- **Database:** SQLite with SQLAlchemy ORM
- **Computer Vision & AI:**
  - OpenCV 4.10 (Image processing and geometry)
  - MediaPipe (Face detection and FaceMesh for head pose estimation)
  - Ultralytics YOLOv8 (Real-time object detection)
- **Reporting:** fpdf2 (PDF report generation)
- **Authentication:** Flask-Login, bcrypt, PyJWT

## CHAPTER 5: ARCHITECTURE AND DESIGN
The ExamSentinelX AI system employs a modular, client-server architecture. The frontend captures the student's webcam feed and transmits base64-encoded frames to the backend via WebSockets.

### Backend Infrastructure and Database Design
The backend manages the SQLite database (via SQLAlchemy) for storing user data, exam details, and violation logs. Key entities include:
- `User`: Manages authentication for students and admins.
- `Exam` & `Question`: Stores exam configurations and multiple-choice questions.
- `ExamSession`: The core table tracking a student's exam attempt, total violations, and calculated risk score (0.0 to 10.0).
- `ViolationLog`: Records specific cheating events, AI confidence scores, timestamps, and evidence screenshots paths.

### Detection Pipeline Architecture
The AI pipeline processes incoming frames sequentially through predefined layers:
1. **Face Detection:** Verifies face presence or multiple faces.
2. **Head Pose Estimation:** Calculates Pitch, Yaw, and Roll to detect looking away.
3. **Object Detection:** Identifies prohibited items like phones and laptops.
4. **Behavioral Classification:** A CNN ensemble to classify overall suspicious behavior.
5. **Lip Movement Detection:** Analyzes Mouth Aspect Ratio (MAR) to detect talking.

## CHAPTER 6: IMPLEMENTATION
### 6.1 Keyframe Extraction
To optimize CNN training and reduce computational load, a motion-based keyframe extraction algorithm was implemented. It calculates the absolute pixel difference between consecutive frames. A frame is selected as a keyframe if the average difference exceeds a calculated threshold: `T = mean(absdiff) + std(absdiff)`. An optimal threshold of 340,000 with a skip factor of 3 is used.

### 6.2 AI Detection Layers Implementation
- **Layer 1 (MediaPipe Face):** Rapidly detects if a face is absent (threshold: 3 seconds) or if multiple persons are present.
- **Layer 2 (MediaPipe + OpenCV):** `solvePnP` calculates the rotation matrix. If absolute Yaw > 30 degrees or Pitch > 20 degrees, a `looking_away` violation is flagged.
- **Layer 3 (YOLOv8):** Detects class 67 (cell phone) for a `phone_detected` violation, or other classes (e.g., books) for a `suspicious_object` violation.
- **Layer 4 (CNN Ensemble):** Integrating InceptionV3 and DenseNet121, it requires two consecutive frames flagged with a confidence >= 0.70 to trigger a `suspicious_behavior` violation.
- **Layer 5 (TalkingDetector):** If the MAR > 0.25 for a sustained duration (e.g., 5 frames), it logs a `talking_to_others` violation.

### 6.3 Priority Logic
To avoid redundant logging, violations are prioritized:
1. `phone_detected` (Highest)
2. `multiple_persons`
3. `suspicious_object`
4. `face_absent`
5. `looking_away` (Lowest)

## CHAPTER 7: RESULTS AND DISCUSSION
The performance of the implemented models was evaluated based on the criteria established in the referenced IEEE research. The system demonstrated robust real-time performance within a web browser context.

### 7.1 YOLOv8 Performance
The implementation upgraded the paper's proposed YOLOv5 to YOLOv8.
- **YOLOv8n** achieves an mAP@0.5 of 37.3 on the COCO dataset, compared to YOLOv5n's 28.0.
- This results in approximately twice the inference speed, enabling efficient real-time processing of frames every two seconds without dropping frames.
- The fallback to the COCO pretrained YOLOv8n model demonstrated highly accurate detection of key prohibited items (phones, persons) with minimal latency.

### 7.2 Deep Learning Architectures Evaluation
Based on evaluations mirroring the reference paper:
- **YOLOv8:** Demonstrated the highest accuracy (Precision ~95.5%, mAP >95%).
- **Inception_ResNet_v2 & Inception-V3:** Showed precision in the mid-80s (~86-87%).
- **Custom CNN:** Achieved precision of ~80%.
The modular integration of these architectures ensures the system can dynamically leverage the most confident models for classification tasks.

## CHAPTER 8: CHALLENGES AND LEARNING OUTCOMES
### Challenges Encountered:
1. **Computational Overhead:** Running multiple deep learning models concurrently (YOLOv8, MediaPipe, CNNs) is resource-intensive. This was mitigated by running inference every 2 seconds rather than on every frame, and prioritizing lightweight geometry-based checks (MediaPipe) before heavier CNN inference.
2. **Browser Compatibility:** Maintaining continuous video frame transmission required stable WebSocket connections. Deprecated API patterns (`AbortSignal.timeout`) caused browser crashes and were resolved by implementing a robust `AbortController` with a `setTimeout` fallback.
3. **False Positives:** Transient movements or yawning frequently triggered false positive cheating flags. Implementing frame-duration thresholds (e.g., 5 frames for talking, 3 seconds for face absence) resolved this issue.

### Learning Outcomes:
- Deepened understanding of optimizing computer vision pipelines for real-time web applications.
- Gained practical experience in integrating multiple deep learning architectures (YOLO, Inception, DenseNet) into a unified system.
- Mastered advanced web socket communication for real-time AI monitoring and asynchronous processing in Flask.

## CHAPTER 9: FUTURE ENHANCEMENTS
To further enhance the system, the following improvements are proposed:
- **Eye Gaze Tracking:** Integrating precise eye-tracking algorithms to detect instances where a student is looking off-screen without significant head movement.
- **Audio Analysis:** Implementing audio monitoring (e.g., Whisper AI) to detect whispers, background conversations, or audio-based cheating that bypasses visual detection.
- **Screen Activity Monitoring:** Adding capabilities to detect tab-switching or the use of unauthorized software on the student's device.
- **Identity Verification:** Implementing facial recognition at the start of the session to ensure the registered student is the one taking the exam.
- **Deployment & Scalability:** Containerizing the application using Docker and migrating to a robust production database like PostgreSQL for large-scale university deployments.

## CHAPTER 10: CONCLUSION
ExamSentinelX AI successfully demonstrates the viability of integrating advanced computer vision and deep learning techniques into a web-based examination platform to detect cheating activities. By combining MediaPipe's geometric facial analysis with YOLOv8's rapid object detection and CNN-based behavioral classification, the system provides a robust, multi-layered approach to online proctoring. The application accurately identifies the four primary cheating classes identified in contemporary research: external devices, head movement, multiple persons, and talking. The modular, scalable architecture provides a solid foundation for future enhancements, paving the way for fully automated, highly secure online assessment environments.
