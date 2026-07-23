"""
ai/head_pose_estimator.py — Head Pose Detection Module
====================================================================
PURPOSE: Estimate student's head orientation to detect "looking away" behavior.

Uses MediaPipe FaceLandmarker to get the facial transformation matrix,
which provides highly accurate 3D rotation (yaw, pitch, roll) even when 
the head stays in the center of the frame.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HeadPoseResult:
    """Structured result from head pose estimation."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    direction: str = 'forward'
    violation_type: Optional[str] = None
    confidence: float = 0.0
    landmarks_found: bool = False
    annotated_frame: Optional[np.ndarray] = None


class HeadPoseEstimator:
    """
    Head pose estimator using MediaPipe FaceLandmarker (Tasks API).
    Determines if student is looking away from screen.
    """

    def __init__(self,
                 yaw_limit: int = 30,
                 pitch_limit: int = 20,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        self.yaw_limit = yaw_limit
        self.pitch_limit = pitch_limit

        self._face_landmarker = None
        self._try_init_mediapipe()

        logger.info("HeadPoseEstimator initialized")

    def _try_init_mediapipe(self) -> bool:
        try:
            import mediapipe as mp
            search_paths = [
                Path(__file__).parent / 'models' / 'face_landmarker.task',
                Path(__file__).parent / 'models' / 'face_landmarker_v2_with_blendshapes.task',
            ]
            model_path = next((p for p in search_paths if p.exists()), None)
            if not model_path:
                return False

            opts = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                output_facial_transformation_matrixes=True
            )
            self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
            return True
        except Exception as e:
            logger.warning(f"HeadPose MediaPipe init failed: {e}")
            return False

    def estimate(self, frame: np.ndarray, draw_annotations: bool = True) -> HeadPoseResult:
        result = HeadPoseResult()
        annotated = frame.copy() if draw_annotations else None
        
        if frame is None or self._face_landmarker is None:
            result.annotated_frame = annotated
            return result

        h, w = frame.shape[:2]
        
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = self._face_landmarker.detect(mp_img)

            if not det.face_landmarks or not det.facial_transformation_matrixes:
                result.annotated_frame = annotated
                return result

            # Extract rotation from transformation matrix
            matrix = det.facial_transformation_matrixes[0]
            # Rotation matrix is top-left 3x3
            r_mat = matrix[:3, :3]
            
            # Convert rotation matrix to Euler angles (yaw, pitch, roll)
            sy = np.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])
            singular = sy < 1e-6

            import math
            if not singular:
                x = math.atan2(r_mat[2, 1], r_mat[2, 2])
                y = math.atan2(-r_mat[2, 0], sy)
                z = math.atan2(r_mat[1, 0], r_mat[0, 0])
            else:
                x = math.atan2(-r_mat[1, 2], r_mat[1, 1])
                y = math.atan2(-r_mat[2, 0], sy)
                z = 0

            # Convert to degrees
            pitch = np.degrees(x)
            yaw = -np.degrees(y)
            roll = np.degrees(z)

            result.landmarks_found = True
            result.yaw = yaw
            result.pitch = pitch
            result.roll = roll

            # Classify direction
            result.direction = self._classify_direction(yaw, pitch)

            # Check violation
            abs_yaw = abs(yaw)
            abs_pitch = abs(pitch)
            if abs_yaw > self.yaw_limit or abs_pitch > self.pitch_limit:
                result.violation_type = 'looking_away'
                max_dev = max(abs_yaw / self.yaw_limit, abs_pitch / self.pitch_limit)
                result.confidence = min(0.99, 0.5 + (max_dev - 1.0) * 0.3)

            # Draw annotations
            if draw_annotations and annotated is not None:
                # Get nose tip for drawing
                nose = det.face_landmarks[0][1] # nose tip
                nx, ny = int(nose.x * w), int(nose.y * h)
                self._draw_pose(annotated, result, (nx, ny))

            result.annotated_frame = annotated
            return result
            
        except Exception as e:
            logger.error(f"Pose error: {e}")
            result.annotated_frame = annotated
            return result

    def _classify_direction(self, yaw: float, pitch: float) -> str:
        if abs(yaw) > self.yaw_limit:
            return 'left' if yaw < 0 else 'right'
        if pitch < -self.pitch_limit:
            return 'down'
        if pitch > self.pitch_limit:
            return 'up'
        return 'forward'

    def _draw_pose(self, frame, result, center):
        color = (0, 0, 255) if result.violation_type else (0, 255, 200)

        # Direction arrow from nose
        arrow_len = 60
        dx = int(np.sin(np.radians(result.yaw)) * arrow_len)
        dy = int(-np.sin(np.radians(result.pitch)) * arrow_len)
        cv2.arrowedLine(frame, center, (center[0] + dx, center[1] + dy), color, 3, tipLength=0.3)

        # Text overlay
        direction_text = f'{result.direction.upper()}'
        angle_text = f'Y:{result.yaw:.0f} P:{result.pitch:.0f}'

        cv2.putText(frame, direction_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, angle_text, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if result.violation_type:
            cv2.putText(frame, 'LOOKING AWAY',
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def close(self):
        if self._face_landmarker:
            self._face_landmarker.close()
