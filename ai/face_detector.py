"""
ai/face_detector.py — Face Detection Module
=========================================================
PURPOSE: Detect faces in video frames.

Uses MediaPipe FaceDetection (which robustly handles side profiles, 
head turning, and partial occlusions) as the primary engine.
Falls back to OpenCV Haar Cascade only if MediaPipe is unavailable.

VIOLATIONS TRIGGERED:
    - face_absent: 0 faces detected
    - multiple_persons: More than 1 face detected
"""

import cv2
import cv2.data
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionResult:
    """Structured result from face detection."""
    face_count: int = 0
    faces: List[dict] = field(default_factory=list)
    violation_type: Optional[str] = None
    confidence: float = 0.0
    annotated_frame: Optional[np.ndarray] = None


class FaceDetector:
    """
    Face detector for exam proctoring using MediaPipe.
    Uses MediaPipe Tasks API FaceLandmarker to bypass deprecated mp.solutions.
    """

    def __init__(self, min_detection_confidence: float = 0.6):
        self.min_confidence = min_detection_confidence
        self._mode = 'disabled'
        self.mp_detector = None

        try:
            import mediapipe as mp
            from pathlib import Path
            
            # Use the existing face_landmarker.task model
            model_path = Path(__file__).parent / 'models' / 'face_landmarker.task'
            if model_path.exists():
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
                BaseOptions = mp.tasks.BaseOptions
                RunningMode = mp.tasks.vision.RunningMode

                opts = FaceLandmarkerOpts(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.IMAGE,
                    num_faces=3, # Need to detect multiple persons
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_detection_confidence
                )
                self.mp_detector = FaceLandmarker.create_from_options(opts)
                self._mode = 'mediapipe_tasks'
                logger.info("Using MediaPipe Tasks API for face detection")
            else:
                logger.error("face_landmarker.task model not found in ai/models/")
        except Exception as e:
            logger.error(f"MediaPipe FaceLandmarker init failed: {e}")

    def detect(self, frame: np.ndarray, draw_annotations: bool = True) -> FaceDetectionResult:
        if frame is None or self._mode == 'disabled':
            return FaceDetectionResult(violation_type='face_absent', confidence=1.0)

        result = FaceDetectionResult()
        annotated = frame.copy() if draw_annotations else None
        h, w = frame.shape[:2]

        faces = self._detect_mediapipe_tasks(frame, w, h)

        result.face_count = len(faces)
        result.faces = faces

        # Draw annotations
        if draw_annotations and annotated is not None:
            for face in faces:
                self._draw_face_box(annotated, face, face.get('confidence', 0.9))

        # Determine violation
        if result.face_count == 0:
            result.violation_type = 'face_absent'
            result.confidence = 0.99
        elif result.face_count > 1:
            result.violation_type = 'multiple_persons'
            result.confidence = 0.99

        result.annotated_frame = annotated
        return result

    def _detect_mediapipe_tasks(self, frame: np.ndarray, w: int, h: int) -> List[dict]:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        try:
            results = self.mp_detector.detect(mp_img)
        except Exception as e:
            logger.error(f"FaceLandmarker detect error: {e}")
            return []

        faces = []
        if results.face_landmarks:
            for lm in results.face_landmarks:
                # Calculate bounding box from landmarks
                x_coords = [p.x * w for p in lm]
                y_coords = [p.y * h for p in lm]
                
                xmin = int(min(x_coords))
                ymin = int(min(y_coords))
                xmax = int(max(x_coords))
                ymax = int(max(y_coords))
                
                # Constrain to frame
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(xmax, w)
                ymax = min(ymax, h)
                
                faces.append({
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'confidence': 1.0
                })
        return faces

    def _draw_face_box(self, frame: np.ndarray, face: dict, confidence: float):
        x, y, fw, fh = face['bbox']
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        label = f"Face {confidence:.2f}"
        cv2.putText(frame, label, (x, max(10, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def close(self):
        if self.mp_detector:
            try:
                self.mp_detector.close()
            except:
                pass
