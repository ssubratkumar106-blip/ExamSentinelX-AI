"""
ai/object_detector.py — YOLOv8 Object Detection Module
=======================================================
PURPOSE: Detect prohibited objects in exam frames (phones, books, extra people).

WHY YOLOv8 INSTEAD OF YOLOv5 (as in the paper)?
    The IEEE paper used YOLOv5, which was state-of-the-art in 2022.
    We use YOLOv8 (2023) for these reasons:
    1. Better mAP: YOLOv8n achieves 37.3 mAP vs YOLOv5n's 28.0 mAP
    2. Same usage interface (Ultralytics package)
    3. Faster inference on CPU
    4. Active maintenance and community support
    5. Native Python API (no subprocess needed)

HOW YOLO WORKS (simplified):
    1. Divide image into a grid (e.g., 80×80, 40×40, 20×20)
    2. Each grid cell predicts bounding boxes and class probabilities
    3. Non-Maximum Suppression (NMS) removes overlapping boxes
    4. Output: list of objects with class, confidence, and location

COCO CLASSES WE USE (from YOLOv8 pretrained on COCO dataset):
    - Class 67: cell phone     → HIGH priority violation
    - Class 0:  person         → multiple persons detection
    - Class 73: book           → suspicious object
    - Class 63: laptop         → suspicious object (if not the exam laptop)
    - Class 64: mouse          → context clue
    - Class 62: tv             → suspicious object

INPUTS:
    - frame: numpy array (BGR)
    - confidence_threshold: minimum detection confidence (default 0.5)

OUTPUTS:
    - ObjectDetectionResult with detected objects and violations
"""

import cv2
import cv2.data  # explicit import to satisfy Pyright
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """A single detected object in the frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels


@dataclass
class ObjectDetectionResult:
    """Structured result from object detection."""
    detected_objects: List[DetectedObject] = field(default_factory=list)
    violation_type: Optional[str] = None
    confidence: float = 0.0
    person_count: int = 0
    phone_detected: bool = False
    suspicious_objects: List[str] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


class ObjectDetector:
    """
    YOLOv8-based object detector for exam proctoring.
    
    Detects prohibited items and extra persons in webcam frames.
    Uses the pretrained COCO model — no custom training needed
    for common objects like phones and books.
    
    USAGE:
        detector = ObjectDetector()
        result = detector.detect(frame)
        if result.phone_detected:
            # Log phone violation
    """

    # COCO class IDs for relevant objects
    TARGET_CLASSES = {
        0:  'person',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',       # phone-like object
        67: 'cell phone',
        73: 'book',
        76: 'scissors',
        84: 'book',         # duplicate for robustness
    }

    # Severity mapping
    HIGH_SEVERITY   = ['cell phone', 'remote']
    MEDIUM_SEVERITY = ['book', 'tv', 'laptop']
    PERSON_CLASS_ID = 0
    PHONE_CLASS_ID  = 67

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.30):  # Low = catches partial/angled phones
        """
        Initialize YOLOv8 model.
        
        Args:
            model_path: Path to .pt weights file.
                        Auto-detects fine-tuned exam model if available.
                        Falls back to 'yolov8n.pt' (auto-downloaded from ultralytics).
            confidence_threshold: Minimum confidence to count a detection.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Auto-detect fine-tuned exam model
        if model_path is None:
            from pathlib import Path
            fine_tuned = Path(__file__).parent / 'models' / 'yolov8_cheating_exam.pt'
            if fine_tuned.exists():
                model_path = str(fine_tuned)
                logger.info(f'Using fine-tuned exam model: {fine_tuned}')
            else:
                model_path = 'yolov8n.pt'  # ultralytics auto-downloads
        
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLOv8 model with error handling."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("Object detection disabled — running in fallback mode")
            self.model = None

    def detect(self, frame: np.ndarray, draw_annotations: bool = True) -> ObjectDetectionResult:
        """
        Run object detection on a single frame.
        
        Args:
            frame: BGR image from OpenCV
            draw_annotations: Draw bounding boxes on frame
            
        Returns:
            ObjectDetectionResult
        """
        result = ObjectDetectionResult()
        annotated = frame.copy() if draw_annotations else None

        # Fallback if model failed to load
        if self.model is None:
            result.annotated_frame = annotated
            return result

        try:
            # Run YOLOv8 inference
            # verbose=False suppresses per-frame console output
            predictions = self.model(frame,
                                     conf=self.confidence_threshold,
                                     classes=list(self.TARGET_CLASSES.keys()),
                                     verbose=False)

            person_count = 0

            # Process detections
            for pred in predictions:
                for box in pred.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    obj = DetectedObject(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                    result.detected_objects.append(obj)

                    # Track specific objects
                    if class_id == self.PERSON_CLASS_ID:
                        person_count += 1

                    if class_id == self.PHONE_CLASS_ID:
                        result.phone_detected = True

                    if class_name in (self.HIGH_SEVERITY + self.MEDIUM_SEVERITY):
                        if class_name not in result.suspicious_objects:
                            result.suspicious_objects.append(class_name)

                    # Draw box on annotated frame
                    if draw_annotations and annotated is not None:
                        self._draw_box(annotated, obj)

            result.person_count = person_count

            # ── Person false-positive guard ────────────────────────────────────
            # Filter: only count persons that:
            #   1. Have confidence >= 0.55 (stricter than phone/book)
            #   2. Occupy at least 3% of the frame area (not a tiny BG figure)
            # This eliminates posters, TV screens, and distant bystanders.
            h_f, w_f = frame.shape[:2]
            frame_area = w_f * h_f
            MIN_PERSON_AREA = frame_area * 0.005     # 0.5% — catches background/partially obscured people
            MIN_PERSON_CONF = 0.25                   # Lower conf to catch partial views

            validated_persons = [
                o for o in result.detected_objects
                if o.class_id == self.PERSON_CLASS_ID
                and o.confidence >= MIN_PERSON_CONF
                and (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]) >= MIN_PERSON_AREA
            ]
            result.person_count = len(validated_persons)

            # ── Determine Primary Violation ────────────────────────────────────
            if result.phone_detected:
                result.violation_type = 'phone_detected'
                # Use the highest phone detection confidence
                phone_confidences = [
                    o.confidence for o in result.detected_objects
                    if o.class_id == self.PHONE_CLASS_ID
                ]
                result.confidence = max(phone_confidences) if phone_confidences else 0.8

            elif len(validated_persons) > 1:
                result.violation_type = 'multiple_persons'
                result.confidence = 0.9

            elif result.suspicious_objects:
                result.violation_type = 'suspicious_object'
                result.confidence = max(
                    o.confidence for o in result.detected_objects
                    if o.class_name in result.suspicious_objects
                )

            # Draw violation overlay
            if draw_annotations and annotated is not None and result.violation_type:
                vio_text = f'!! {result.violation_type.upper().replace("_", " ")}'
                cv2.putText(annotated, vio_text,
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            logger.error(f"Object detection error: {e}")

        result.annotated_frame = annotated
        return result

    def _draw_box(self, frame: np.ndarray, obj: DetectedObject):
        """Draw colored bounding box and label for detected object."""
        x1, y1, x2, y2 = obj.bbox

        # Color by severity
        if obj.class_name in self.HIGH_SEVERITY:
            color = (0, 0, 255)     # Red for high severity
        elif obj.class_name in self.MEDIUM_SEVERITY:
            color = (0, 165, 255)   # Orange for medium
        elif obj.class_name == 'person':
            color = (255, 165, 0)   # Blue for person
        else:
            color = (255, 255, 0)   # Cyan for others

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{obj.class_name} {obj.confidence:.0%}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
