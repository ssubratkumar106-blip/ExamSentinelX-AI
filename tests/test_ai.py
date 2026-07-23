"""
tests/test_ai.py — AI Module Tests
=====================================
Tests for the face detector, head pose, and object detector modules.
"""

import pytest
import numpy as np
import cv2


class TestFaceDetector:
    """Tests for the FaceDetector module."""

    def test_import(self):
        """Verify FaceDetector can be imported."""
        from ai.face_detector import FaceDetector, FaceDetectionResult
        assert FaceDetector is not None

    def test_result_no_frame(self):
        """Passing None frame should return face_absent violation."""
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        result = detector.detect(None, draw_annotations=False)
        assert result.violation_type == 'face_absent'
        assert result.face_count == 0
        detector.close()

    def test_black_frame(self):
        """Solid black frame should have no faces detected."""
        from ai.face_detector import FaceDetector
        detector = FaceDetector()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(black_frame, draw_annotations=False)
        assert result.face_count == 0
        detector.close()


class TestHeadPoseEstimator:
    """Tests for the HeadPoseEstimator module."""

    def test_import(self):
        """Verify HeadPoseEstimator can be imported."""
        from ai.head_pose_estimator import HeadPoseEstimator, HeadPoseResult
        assert HeadPoseEstimator is not None

    def test_no_landmarks(self):
        """Black frame with no face should return no landmarks."""
        from ai.head_pose_estimator import HeadPoseEstimator
        estimator = HeadPoseEstimator()
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(black_frame, draw_annotations=False)
        assert result.landmarks_found == False
        assert result.violation_type is None
        estimator.close()


class TestObjectDetector:
    """Tests for the ObjectDetector module."""

    def test_import(self):
        """Verify ObjectDetector can be imported."""
        from ai.object_detector import ObjectDetector, ObjectDetectionResult
        assert ObjectDetector is not None

    def test_empty_frame(self):
        """Random noise frame should not crash."""
        from ai.object_detector import ObjectDetector
        detector = ObjectDetector()
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(noise, draw_annotations=False)
        assert result is not None
        assert isinstance(result.detected_objects, list)


class TestAnalysisResult:
    """Tests for the AnalysisResult serialization."""

    def test_to_dict(self):
        """AnalysisResult should serialize to JSON-compatible dict."""
        from ai.detector import AnalysisResult
        result = AnalysisResult(
            timestamp='2024-01-01T00:00:00',
            session_id=1,
            has_violation=True,
            violation_type='phone_detected',
            confidence=0.87
        )
        d = result.to_dict()
        assert d['has_violation'] == True
        assert d['violation_type'] == 'phone_detected'
        assert d['confidence'] == 0.87


# ── Run tests ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
