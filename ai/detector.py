"""
ai/detector.py — Main Detection Orchestrator
=============================================
PURPOSE: Combine all AI modules into a single detection pipeline.

DETECTION LAYERS (from IEEE paper approach):
    Layer 1 — YOLOv8 Object Detection:
        Detects phones, books, extra persons in real-time.
        Uses fine-tuned 'cheating-online-exam' COCO dataset.
        Falls back to pretrained COCO weights if custom model unavailable.

    Layer 2 — MediaPipe Face + Head Pose:
        Detects face absence, multiple faces, head direction.
        Pure geometry-based — no training needed.

    Layer 3 — CNN Ensemble Classifier (InceptionV3 + DenseNet121):
        Classifies overall frame behavior as cheating/normal.
        Loaded from fine-tuned .h5 models when available.
        Falls back gracefully when TensorFlow unavailable.

HOW IT WORKS:
    1. Receive a base64-encoded frame from the browser webcam
    2. Decode it to a numpy array (OpenCV format)
    3. Run all 3 AI layers in sequence
    4. Merge results using priority-based violation selection
    5. Return JSON-serializable result to the backend API
"""

import cv2
import base64
import json
import os
import time
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from pathlib import Path

from ai.face_detector import FaceDetector, FaceDetectionResult
from ai.head_pose_estimator import HeadPoseEstimator, HeadPoseResult
from ai.object_detector import ObjectDetector, ObjectDetectionResult
from ai.cnn_classifier import get_cnn_classifier
from ai.lip_detector import TalkingDetector

import logging
logger = logging.getLogger(__name__)

# ── Auto-detect best available YOLO model ─────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "ai" / "models"

def _get_best_yolo_model() -> str:
    """
    Return path to best available YOLO model:
    1. Fine-tuned exam-specific model (if trained)
    2. COCO pretrained yolov8n.pt (fallback)
    """
    custom = MODELS_DIR / "yolov8_cheating_exam.pt"
    if custom.exists():
        logger.info(f"Using fine-tuned exam model: {custom}")
        return str(custom)
    # Fallback: ultralytics will auto-download yolov8n.pt from internet
    logger.info("Using COCO pretrained YOLOv8n (fine-tuned model not found)")
    logger.info("  To improve: python ai/train_model.py --mode yolo")
    return "yolov8n.pt"


@dataclass
class AnalysisResult:
    """
    Unified result from the complete detection pipeline.
    This is returned to the Flask API as JSON.
    """
    timestamp: str = ''
    session_id: Optional[int] = None
    has_violation: bool = False
    violation_type: Optional[str] = None
    confidence: float = 0.0
    details: Dict = field(default_factory=dict)
    face_count: int = 0
    head_direction: str = 'forward'
    detected_objects: List[str] = field(default_factory=list)
    annotated_frame_b64: Optional[str] = None  # Base64 frame with annotations

    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


class ExamDetector:
    """
    Main proctoring detector that runs the complete AI pipeline.
    
    This is instantiated ONCE per active exam session and reused
    for each frame — this avoids expensive model reloads.
    
    USAGE:
        detector = ExamDetector(session_id=42)
        result = detector.analyze_frame(base64_frame)
        if result.has_violation:
            save_violation_to_db(result)
    """

    def __init__(self,
                 session_id: int,
                 capture_dir: Optional[str] = None,
                 face_absence_threshold: int = 3,
                 yaw_limit: int = 25,    # Tighter: 25° catches natural side glances
                 pitch_limit: int = 18,  # Tighter: 18° catches looking down at phone
                 confidence_threshold: float = 0.35,  # Lower = catches partial views
                 use_cnn: bool = True):
        """
        Initialize all AI sub-detectors.

        Args:
            session_id: Database ID of the current exam session
            capture_dir: Directory to save violation screenshots
            face_absence_threshold: Seconds before flagging face absence
            yaw_limit: Max allowed yaw before flagging looking away
            pitch_limit: Max allowed pitch before flagging
            confidence_threshold: Minimum confidence for YOLO detections
            use_cnn: Also run CNN ensemble classifier (requires TF + trained models)
        """
        self.session_id = session_id
        self.capture_dir = capture_dir or str(BASE_DIR / 'captures' / 'evidence')
        self.face_absence_threshold = face_absence_threshold
        self.use_cnn = use_cnn

        # ── Initialize AI Modules ──────────────────────────────────────────────
        logger.info(f"Initializing AI detectors for session {session_id}")

        # Layer 1: MediaPipe face + head pose
        self.face_detector = FaceDetector(min_detection_confidence=0.3)  # Lower = catches more faces
        self.head_pose = HeadPoseEstimator(
            yaw_limit=yaw_limit,
            pitch_limit=pitch_limit
        )

        # Layer 2: YOLOv8 object detection (auto-selects best model)
        yolo_model = _get_best_yolo_model()
        self.object_detector = ObjectDetector(
            model_path=yolo_model,
            confidence_threshold=confidence_threshold
        )

        # Layer 3: CNN ensemble classifier
        self.cnn_classifier = None
        if use_cnn:
            try:
                self.cnn_classifier = get_cnn_classifier()
                if self.cnn_classifier:
                    logger.info("CNN ensemble classifier loaded")
            except Exception as e:
                logger.warning(f"CNN classifier unavailable: {e}")

        # Layer 4: Lip/Talking detector
        # mar_threshold=0.22: sensitive enough to catch real talking
        # sustained_frames=2: fires after 2 consecutive open frames (3s at 1.5s interval)
        self.talking_detector = TalkingDetector(mar_threshold=0.22, sustained_frames=2)
        self.talking_detector.initialize()

        # ── State Tracking ─────────────────────────────────────────────────────
        self._face_absent_start: Optional[float] = None
        self._last_analysis_time: float = 0
        self._cnn_violation_count: int = 0  # Track consecutive CNN violations

        # Debounce: 1 = fire immediately, 2 = require 2 consecutive frames
        self._multi_person_frames: int = 0
        self._looking_away_frames: int = 0
        self.MULTI_PERSON_THRESHOLD  = 1   # Fire immediately on multi-person
        self.LOOKING_AWAY_THRESHOLD  = 2   # 2 consecutive frames for looking away

        # Ensure capture directory exists
        os.makedirs(self.capture_dir, exist_ok=True)

        logger.info("AI detector initialized successfully")

    def analyze_frame(self, frame_b64: str, return_annotated: bool = True, trust_score: float = 100.0) -> AnalysisResult:
        """
        Analyze one webcam frame for cheating activity.
        
        Args:
            frame_b64: Base64-encoded JPEG frame from browser
            return_annotated: Return annotated frame with bounding boxes
            trust_score: Current trust score to render in the HUD

            
        Returns:
            AnalysisResult with violation information
        """
        result = AnalysisResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id
        )

        # ── Decode Frame ───────────────────────────────────────────────────────
        frame = self._decode_frame(frame_b64)
        if frame is None:
            logger.warning("Failed to decode frame")
            return result

        # ── Layer 1+2: MediaPipe Face + Head Pose + YOLOv8 ───────────────────
        raw_frame = frame.copy()
        final_frame = frame.copy()

        face_result = self.face_detector.detect(raw_frame, draw_annotations=False)
        pose_result = self.head_pose.estimate(raw_frame, draw_annotations=False)
        obj_result = self.object_detector.detect(raw_frame, draw_annotations=False)

        # Draw annotations on the final frame (so inference is not confused by drawn boxes)
        if return_annotated:
            for face in face_result.faces:
                self.face_detector._draw_face_box(final_frame, face, face.get('confidence', 0.9))
            if pose_result.landmarks_found:
                # Find nose tip manually since we didn't use draw_annotations=True
                # But wait, we can't easily access det.face_landmarks outside.
                # It's better to just skip head pose arrows or let it be. Actually pose_result doesn't store the nose point.
                pass
            for obj in obj_result.detected_objects:
                self.object_detector._draw_box(final_frame, obj)

        # ── Fill Basic Fields ──────────────────────────────────────────────────
        result.face_count = face_result.face_count
        result.head_direction = pose_result.direction
        result.detected_objects = [o.class_name for o in obj_result.detected_objects]

        # ── Primary Violation: YOLOv8 + MediaPipe ─────────────────────────────
        # Priority: phone > multiple_persons > face_absent > suspicious_object > looking_away
        violation, confidence, details = self._determine_violation(
            face_result, pose_result, obj_result
        )

        # ── Layer 3: CNN Classifier (secondary confirmation) ───────────────────
        cnn_violation = None
        if self.cnn_classifier and not violation:
            # Only run CNN if primary pipeline didn't already flag a violation
            # (saves compute; CNN is used as a supplementary detector)
            try:
                cnn_result = self.cnn_classifier.classify_frame(frame)
                if cnn_result.is_suspicious and cnn_result.confidence >= 0.7:
                    self._cnn_violation_count += 1
                    # Require 2+ consecutive CNN flags to avoid false positives
                    if self._cnn_violation_count >= 2:
                        cnn_violation = 'suspicious_behavior'
                        violation = cnn_violation
                        confidence = cnn_result.confidence
                        details = {
                            'cnn_model': cnn_result.model_name,
                            'predicted_class': cnn_result.predicted_class,
                            'class_probs': cnn_result.all_probs
                        }
                else:
                    self._cnn_violation_count = max(0, self._cnn_violation_count - 1)

                # Annotate frame with CNN result
                label = f"CNN: {cnn_result.predicted_class} ({cnn_result.confidence:.0%})"
                color = (0, 0, 255) if cnn_result.is_suspicious else (0, 200, 0)
                cv2.putText(final_frame, label,
                            (10, final_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            except Exception as e:
                logger.debug(f"CNN classifier error: {e}")

        if violation:
            result.has_violation = True
            result.violation_type = violation
            result.confidence = confidence
            result.details = details

            # Save screenshot for evidence
            screenshot_path = self._save_screenshot(final_frame, violation)
            result.details['screenshot_path'] = screenshot_path

        # ── Layer 4: Talking Detection ─────────────────────────────────────────
        try:
            talk_result = self.talking_detector.process_frame(frame)
            # Draw MAR visualization on the annotated frame (always)
            if talk_result.get('face_detected'):
                final_frame = self.talking_detector.draw_visualization(final_frame, talk_result)
            if talk_result.get('is_talking') and not result.has_violation:
                result.has_violation = True
                result.violation_type = 'talking_to_others'
                result.confidence = min(0.95, 0.60 + talk_result.get('mar', 0) * 1.2)
                result.details = {
                    'mar': round(talk_result.get('mar', 0), 3),
                    'open_duration_frames': talk_result.get('open_duration_frames', 0),
                }
                screenshot_path = self._save_screenshot(final_frame, 'talking_to_others')
                result.details['screenshot_path'] = screenshot_path
        except Exception as e:
            logger.debug(f"Talking detector error: {e}")

        # -- HUD overlay
        try:
            self._draw_hud(final_frame, result, face_result, pose_result, obj_result, talk_result, trust_score)
        except Exception as e:
            logger.debug(f"HUD draw error: {e}")

        # -- Encode Annotated Frame ─────────────────────────────────────────────
        if return_annotated:
            result.annotated_frame_b64 = self._encode_frame(final_frame)

        self._last_analysis_time = time.time()
        return result

    def _determine_violation(self, face: FaceDetectionResult,
                              pose: HeadPoseResult,
                              obj: ObjectDetectionResult):
        """
        Determine the primary violation with priority ordering.
        
        Returns: (violation_type, confidence, details_dict)
        """
        now = time.time()

        # ── Priority 1: Phone Detected ─────────────────────────────────────────
        if obj.phone_detected:
            return 'phone_detected', obj.confidence, {
                'detected_objects': obj.suspicious_objects,
                'person_count': obj.person_count
            }

        # ── Priority 2: Multiple Persons (debounced — require 3 consecutive frames) ─
        if face.face_count > 1 or obj.person_count > 1:
            self._multi_person_frames += 1
            if self._multi_person_frames >= self.MULTI_PERSON_THRESHOLD:
                face_count = max(face.face_count, obj.person_count)
                return 'multiple_persons', 0.90, {
                    'face_count': face_count,
                    'method': 'face+yolo',
                    'consecutive_frames': self._multi_person_frames
                }
            # Not enough consecutive frames yet — don't flag
        else:
            self._multi_person_frames = 0  # Reset on any frame without multiple persons

        # ── Priority 3: Suspicious Object ─────────────────────────────────────
        if obj.suspicious_objects:
            return 'suspicious_object', obj.confidence, {
                'objects': obj.suspicious_objects
            }

        # ── Priority 4: Face Absence (with time threshold) ────────────────────
        if face.face_count == 0:
            if self._face_absent_start is None:
                self._face_absent_start = now  # Start timer

            elapsed = now - self._face_absent_start
            if elapsed >= self.face_absence_threshold:
                return 'face_absent', 1.0, {
                    'duration_seconds': round(elapsed, 1)
                }
        else:
            self._face_absent_start = None  # Reset timer when face returns

        # ── Priority 5: Looking Away (debounced — require 5 consecutive frames) ─
        if pose.violation_type == 'looking_away':
            self._looking_away_frames += 1
            if self._looking_away_frames >= self.LOOKING_AWAY_THRESHOLD:
                return 'looking_away', pose.confidence, {
                    'yaw': round(pose.yaw, 1),
                    'pitch': round(pose.pitch, 1),
                    'direction': pose.direction
                }
        else:
            self._looking_away_frames = 0  # Reset when back to forward

        return None, 0.0, {}

    def _decode_frame(self, frame_b64: str) -> Optional[np.ndarray]:
        """Decode base64 JPEG string to numpy array."""
        try:
            # Remove data URL prefix if present
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',')[1]

            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None

    def _encode_frame(self, frame: np.ndarray) -> Optional[str]:
        """Encode numpy array to base64 JPEG string."""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return 'data:image/jpeg;base64,' + base64.b64encode(buffer.tobytes()).decode('utf-8')
        except Exception as e:
            logger.error(f"Frame encode error: {e}")
            return None

    def _save_screenshot(self, frame: np.ndarray, violation_type: str) -> Optional[str]:
        """Save violation frame as a timestamped JPEG file."""
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            filename = f"session_{self.session_id}_{violation_type}_{timestamp}.jpg"
            filepath = os.path.join(self.capture_dir, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            logger.error(f"Screenshot save error: {e}")
            return None


    def _draw_hud(self, frame, result, face_result, pose_result, obj_result, talk_result, trust_score):
        """Draw status HUD on annotated frame visible in browser AI view."""
        h, w = frame.shape[:2]
        
        GREEN   = (0, 220, 100)
        RED     = (0, 0, 255)
        CYAN    = (255, 220, 0)
        ORANGE  = (0, 165, 255)
        WHITE   = (255, 255, 255)
        GREY    = (160, 160, 160)
        BG_DARK = (20, 20, 25)

        # ── Top status bar ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 50), BG_DARK, -1)
        cv2.putText(frame, "ExamSentinelX AI", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, CYAN, 2)
        cv2.putText(frame, "ALL DETECTORS TEST", (280, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)

        # ── Right-side panel (status of each detector) ──────────────────────────
        panel_w = 310
        panel_x = max(w - panel_w - 10, 0)
        panel_y = 60
        panel_h = 280
        
        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), BG_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), CYAN, 1)

        y = panel_y + 25
        line_h = 32

        # Header
        cv2.putText(frame, "DETECTION STATUS", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1)
        y += line_h + 5

        # 1. Face Detection
        face_count = face_result.face_count if face_result else 0
        face_ok = face_count >= 1
        face_color = GREEN if face_ok else RED
        face_text = f"{face_count} face(s)" if face_ok else "NO FACE"
        cv2.putText(frame, "FACE:", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, face_text, (panel_x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        y += line_h

        # 2. Head Pose
        direction = pose_result.direction if pose_result else "N/A"
        pose_violation = pose_result.violation_type if pose_result else None
        pose_color = RED if pose_violation else GREEN
        yaw_val = pose_result.yaw if pose_result else 0
        pitch_val = pose_result.pitch if pose_result else 0
        pose_text = f"{direction.upper()} (Y:{yaw_val:.0f} P:{pitch_val:.0f})"
        cv2.putText(frame, "POSE:", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, pose_text, (panel_x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 2)
        y += line_h

        # 3. Object Detection
        phone = obj_result.phone_detected if obj_result else False
        persons = obj_result.person_count if obj_result else 0
        suspicious = obj_result.suspicious_objects if obj_result else []
        if phone:
            obj_text = "PHONE DETECTED!"
            obj_color = RED
        elif persons > 1:
            obj_text = f"MULTI-PERSON ({persons})"
            obj_color = RED
        elif suspicious:
            obj_text = f"SUSPICIOUS: {', '.join(suspicious)}"
            obj_color = ORANGE
        else:
            obj_text = f"CLEAR ({persons} person)"
            obj_color = GREEN
        cv2.putText(frame, "YOLO:", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, obj_text, (panel_x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)
        y += line_h

        # 4. Talking Detection
        is_talking = talk_result.get('is_talking', False) if talk_result else False
        mar = talk_result.get('mar', 0.0) if talk_result else 0.0
        talk_color = RED if is_talking else GREEN
        talk_text = f"TALKING (MAR:{mar:.2f})" if is_talking else f"SILENT (MAR:{mar:.2f})"
        cv2.putText(frame, "TALK:", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, talk_text, (panel_x + 100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, talk_color, 2)
        y += line_h

        # 5. Overall status
        y += 10
        cv2.line(frame, (panel_x + 10, y - 15), (panel_x + panel_w - 10, y - 15), GREY, 1)
        
        # Draw Trust Score
        y += 5
        ts_color = GREEN if trust_score > 80 else (ORANGE if trust_score > 50 else RED)
        cv2.putText(frame, f"TRUST SCORE: {trust_score:.1f}%", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ts_color, 2)
        y += 25

        violations = []
        if not face_ok: violations.append("NO FACE")
        if phone: violations.append("PHONE")
        if persons > 1: violations.append("MULTI-PERSON")
        if pose_violation: violations.append("LOOKING AWAY")
        if is_talking: violations.append("TALKING")
        if suspicious: violations.append("SUSPICIOUS OBJ")

        if violations:
            overall = " | ".join(violations)
            cv2.putText(frame, "ALERT:", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            if len(overall) > 25:
                cv2.putText(frame, overall[:25], (panel_x + 90, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
                cv2.putText(frame, overall[25:], (panel_x + 90, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
            else:
                cv2.putText(frame, overall, (panel_x + 90, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
        else:
            cv2.putText(frame, "STATUS: ALL CLEAR", (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

        # ── Red alert border when any violation ──────────────────────────────────
        if violations or result.has_violation:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 4)

    def cleanup(self):
        """Release all AI model resources."""
        try:
            self.face_detector.close()
            self.head_pose.close()
            logger.info(f"Detector cleanup complete for session {self.session_id}")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# ── Global Detector Registry ───────────────────────────────────────────────────
# Stores active detectors per session. Avoids re-creating models per request.
_active_detectors: Dict[int, ExamDetector] = {}


def get_or_create_detector(session_id: int, **kwargs) -> ExamDetector:
    """Get existing detector for session, or create a new one."""
    if session_id not in _active_detectors:
        _active_detectors[session_id] = ExamDetector(session_id=session_id, **kwargs)
    return _active_detectors[session_id]


def cleanup_detector(session_id: int):
    """Clean up and remove detector when session ends."""
    if session_id in _active_detectors:
        _active_detectors[session_id].cleanup()
        del _active_detectors[session_id]
