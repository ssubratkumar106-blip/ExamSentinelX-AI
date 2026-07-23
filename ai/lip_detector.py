"""
ai/lip_detector.py
==================
Talking Detection — MediaPipe 0.10.x Tasks API + OpenCV Haar Fallback
Paper: Class 4 — "Talking to Others" (talking_to_others)

Detects if a student is talking (mouth open) by measuring the
Mouth Aspect Ratio (MAR) from facial landmark coordinates.

IMPORTANT: MediaPipe 0.10.x removed mp.solutions — this file uses the new
mp.tasks.vision.FaceLandmarker API. Falls back to Haar+geometry if not available.
"""

import cv2
import cv2.data  # explicit import to satisfy Pyright
import numpy as np
from typing import TYPE_CHECKING, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# ── Guard against ultralytics monkey-patching cv2.imshow ─────────────────────
# Capture the real imshow pointer now, before any ultralytics import can
# replace it with a broken unicode_escape wrapper.
_REAL_IMSHOW = cv2.imshow

def _safe_imshow(winname: str, mat):
    safe = winname.encode('ascii', errors='replace').decode('ascii')
    _REAL_IMSHOW(safe, mat)

# ── Config ────────────────────────────────────────────────────────────────────
MAR_THRESHOLD    = 0.28   # MAR above this = mouth open (range 0.0–0.6+)
SUSTAINED_FRAMES = 5      # Consecutive open frames before "talking" fires

# MediaPipe face-mesh lip landmark indices (same numbering as old solutions API)
LIP_TOP    = 13    # Upper inner lip center
LIP_BOTTOM = 14    # Lower inner lip center
LIP_LEFT   = 78    # Left mouth corner
LIP_RIGHT  = 308   # Right mouth corner

# Outer lip indices for polygon visualization
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


class LipDetector:
    """
    Computes Mouth Aspect Ratio (MAR) and tracks talking state.

    MAR = vertical_mouth_opening / horizontal_mouth_width
    If MAR > threshold AND sustained for N frames → talking detected.
    """

    def __init__(self, mar_threshold: float = MAR_THRESHOLD,
                 sustained_frames: int = SUSTAINED_FRAMES):
        self.mar_threshold     = mar_threshold
        self.sustained_frames  = sustained_frames
        self._open_frame_count = 0
        self.is_talking        = False
        self.last_mar          = 0.0
        # Back-reference set by TalkingDetector so reset() can clear motion buffer
        self._parent: Optional['TalkingDetector'] = None

    def compute_mar(self, top, bottom, left, right) -> float:
        """MAR = |top−bottom| / |left−right| from (x,y) pixel coords."""
        vertical   = np.linalg.norm(np.array(bottom) - np.array(top))
        horizontal = np.linalg.norm(np.array(right)  - np.array(left))
        if horizontal < 1e-6:
            return 0.0
        return float(vertical / horizontal)

    def update(self, mar: float) -> dict:
        """Update talking state from a computed MAR value."""
        self.last_mar  = mar
        mouth_open     = mar > self.mar_threshold

        if mouth_open:
            self._open_frame_count += 1
        else:
            self._open_frame_count = max(0, self._open_frame_count - 1)

        self.is_talking = self._open_frame_count >= self.sustained_frames

        return {
            'is_talking':           self.is_talking,
            'mar':                  mar,
            'mouth_open':           mouth_open,
            'open_duration_frames': self._open_frame_count,
        }

    def reset(self):
        """Reset state between exam sessions."""
        self._open_frame_count = 0
        self.is_talking        = False
        self.last_mar          = 0.0
        # Also clear parent's motion buffer if accessible
        if hasattr(self, '_parent') and self._parent is not None:
            self._parent._prev_mouth_gray = None


class TalkingDetector:
    """
    Full pipeline: webcam frame → talking detection.

    Backends (in priority order):
      1. MediaPipe 0.10.x FaceLandmarker Tasks API (requires face_landmarker.task)
      2. OpenCV Haar cascade + geometric mouth-region estimation (always available)
    """

    def __init__(self, mar_threshold: float = MAR_THRESHOLD,
                 sustained_frames: int = SUSTAINED_FRAMES):
        self.lip_detector     = LipDetector(mar_threshold, sustained_frames)
        self.lip_detector._parent = self   # Back-reference for motion buffer reset
        self._mode            = 'disabled'  # 'mediapipe' | 'haar' | 'disabled'
        self._initialized     = False
        self._face_landmarker = None   # MediaPipe FaceLandmarker instance
        self._haar_cascade    = None   # OpenCV Haar cascade
        self._prev_mouth_gray = None   # Previous mouth ROI for motion detection

    # ── Initialization ────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """Try each backend in priority order. Returns True if any succeeds."""
        if self._try_init_mediapipe():
            return True
        logger.info("[LipDetector] MediaPipe not available — using Haar fallback")
        if self._try_init_haar():
            return True
        logger.warning("[LipDetector] No backend available. Lip detection disabled.")
        return False

    def _try_init_mediapipe(self) -> bool:
        """Init MediaPipe 0.10.x Tasks API FaceLandmarker."""
        try:
            import mediapipe as mp  # type: ignore[import]
            from pathlib import Path

            # MediaPipe 0.10.x requires mp.tasks
            if not hasattr(mp, 'tasks') or not hasattr(mp.tasks, 'vision'):
                logger.info("[LipDetector] mp.tasks.vision not found in this mediapipe version")
                return False

            # Look for the face landmarker model file
            search_paths = [
                Path(__file__).parent / 'models' / 'face_landmarker.task',
                Path(__file__).parent / 'models' / 'face_landmarker_v2_with_blendshapes.task',
                Path(__file__).parent.parent / 'face_landmarker.task',
            ]
            model_path = next((p for p in search_paths if p.exists()), None)

            if model_path is None:
                logger.info("[LipDetector] face_landmarker.task not found. "
                            "Download from: https://storage.googleapis.com/mediapipe-models/"
                            "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                            "and place in ai/models/")
                return False

            FaceLandmarker     = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOpts = mp.tasks.vision.FaceLandmarkerOptions
            BaseOptions        = mp.tasks.BaseOptions
            RunningMode        = mp.tasks.vision.RunningMode

            opts = FaceLandmarkerOpts(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._face_landmarker = FaceLandmarker.create_from_options(opts)
            self._mode            = 'mediapipe'
            self._initialized     = True
            logger.info(f"[LipDetector] MediaPipe FaceLandmarker initialized from {model_path}")
            print(f"[LipDetector] MediaPipe FaceLandmarker initialized")
            return True

        except ImportError:
            logger.info("[LipDetector] mediapipe not installed")
            return False
        except Exception as e:
            logger.info(f"[LipDetector] mediapipe init failed: {e}")
            return False

    def _try_init_haar(self) -> bool:
        """Init OpenCV Haar cascade as a lightweight fallback."""
        try:
            cascade_path       = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self._haar_cascade.empty():
                return False
            self._mode        = 'haar'
            self._initialized = True
            logger.info("[LipDetector] Haar cascade initialized (fallback mode)")
            print("[LipDetector] Using Haar cascade for lip detection (fallback mode)")
            return True
        except Exception as e:
            logger.warning(f"[LipDetector] Haar init error: {e}")
            return False

    # ── Frame Processing ──────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Detect talking in a BGR frame.

        Returns:
            {
                'is_talking':           bool,
                'mar':                  float,
                'face_detected':        bool,
                'mouth_open':           bool,
                'open_duration_frames': int,
                'violation_type':       'talking_to_others' | None,
                'mouth_pts':            list of (x,y) tuples for drawing
            }
        """
        empty = {
            'is_talking': False, 'mar': 0.0, 'face_detected': False,
            'mouth_open': False, 'open_duration_frames': 0,
            'violation_type': None, 'mouth_pts': []
        }

        if not self._initialized:
            return empty

        if self._mode == 'mediapipe':
            return self._process_mediapipe(frame_bgr, empty)
        elif self._mode == 'haar':
            return self._process_haar(frame_bgr, empty)
        return empty

    def _process_mediapipe(self, frame_bgr: np.ndarray, empty: dict) -> dict:
        """Run MediaPipe FaceLandmarker and compute MAR from lip points."""
        try:
            import mediapipe as mp  # type: ignore[import]

            h, w = frame_bgr.shape[:2]
            rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            assert self._face_landmarker is not None, "FaceLandmarker not initialised"
            det   = self._face_landmarker.detect(mp_img)

            if not det.face_landmarks:
                self.lip_detector.reset()
                return {**empty, 'face_detected': False}

            lm = det.face_landmarks[0]  # list of NormalizedLandmark

            def pt(idx):
                return (lm[idx].x * w, lm[idx].y * h)

            top    = pt(LIP_TOP)
            bottom = pt(LIP_BOTTOM)
            left   = pt(LIP_LEFT)
            right  = pt(LIP_RIGHT)

            mar    = self.lip_detector.compute_mar(top, bottom, left, right)
            result = self.lip_detector.update(mar)

            # Build mouth polygon for visualization
            mouth_pts = []
            for idx in OUTER_LIP_IDX:
                if idx < len(lm):
                    mouth_pts.append((int(lm[idx].x * w), int(lm[idx].y * h)))

            return {
                **result,
                'face_detected':  True,
                'violation_type': 'talking_to_others' if result['is_talking'] else None,
                'mouth_pts':      mouth_pts,
            }
        except Exception as e:
            logger.debug(f"[LipDetector] mediapipe frame error: {e}")
            return {**empty, 'face_detected': False}

    def _process_haar(self, frame_bgr: np.ndarray, empty: dict) -> dict:
        """
        Estimate mouth openness using Haar face detection + local-variance + motion proxy.

        Strategy (more reliable than dark-pixel ratio):
          1. Detect face with Haar cascade.
          2. Crop lower-mouth ROI (60%–85% vertical of face box).
          3. Compute LOCAL VARIANCE of the ROI as an openness proxy:
             - Closed mouth: uniform skin texture → low variance.
             - Open mouth:   teeth contrast + shadow → high variance.
          4. Additionally compare current ROI with previous frame via absolute
             difference to catch the MOTION of talking (jaw moving).
          5. Combine both scores to form a single MAR-equivalent metric.
        """
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        assert self._haar_cascade is not None, "Haar cascade not initialised"
        faces = self._haar_cascade.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        if len(faces) == 0:
            self.lip_detector.reset()
            self._prev_mouth_gray = None
            return {**empty, 'face_detected': False}

        # Use the largest detected face
        faces = sorted(faces.tolist(), key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Mouth ROI: lower 58%–85% of face box, centred horizontally
        mx  = x + w // 5
        my  = y + int(h * 0.58)
        mw  = w - w // 2 + w // 5   # ~60% of face width
        mh  = int(h * 0.27)
        mouth_roi = gray[my:my + mh, mx:mx + mw]

        if mouth_roi.size == 0:
            return {**empty, 'face_detected': True}

        # ── Metric 1: Local variance (texture contrast) ───────────────────
        # Normalised to [0,1] where 1 = max possible variance (128² for uint8).
        # Empirically: closed mouth ≈ 0.01–0.05, open mouth ≈ 0.08–0.25
        # Cast to float32 ndarray so np.var overload resolves correctly.
        roi_f: np.ndarray = np.asarray(mouth_roi, dtype=np.float32)
        var_score  = float(np.var(roi_f)) / (128.0 ** 2)

        # ── Metric 2: Frame-to-frame motion (jaw / lip movement) ──────────
        motion_score = 0.0
        if self._prev_mouth_gray is not None:
            try:
                prev_resized = cv2.resize(self._prev_mouth_gray, (mouth_roi.shape[1], mouth_roi.shape[0]))
                diff_arr: np.ndarray = np.asarray(
                    cv2.absdiff(mouth_roi, prev_resized), dtype=np.float32
                )
                motion_score = float(np.mean(diff_arr)) / 255.0
            except Exception:
                motion_score = 0.0
        self._prev_mouth_gray = mouth_roi.copy()

        # Combined MAR proxy:
        #   60% variance + 40% motion  — speaking involves BOTH contrast AND motion
        #   Scale factor 1.8 maps typical open-mouth score to ~0.28+ threshold range
        mar = min((var_score * 0.60 + motion_score * 0.40) * 1.8, 0.60)

        result = self.lip_detector.update(mar)
        mouth_pts = [(mx, my), (mx + mw, my), (mx + mw, my + mh), (mx, my + mh)]

        return {
            **result,
            'face_detected':  True,
            'violation_type': 'talking_to_others' if result['is_talking'] else None,
            'mouth_pts':      mouth_pts,
        }

    # ── Visualization ─────────────────────────────────────────────────────────
    def draw_visualization(self, frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw MAR gauge, status text, mouth outline, and alert border on the frame.
        Works for both mediapipe (polygon) and haar (rectangle) modes.
        """
        is_talking = result.get('is_talking', False)
        mar        = result.get('mar', 0.0)
        mouth_open = result.get('mouth_open', False)
        mouth_pts  = result.get('mouth_pts', [])
        h, w = frame_bgr.shape[:2]

        # Color and label based on state
        if is_talking:
            status_color = (0, 0, 220)
            status_text  = "!! TALKING DETECTED !!"
        elif mouth_open:
            status_color = (0, 140, 255)
            status_text  = "Mouth Open"
        else:
            status_color = (0, 210, 60)
            status_text  = "Silent"

        # ── MAR value text ──────────────────────────────────────────────────
        cv2.putText(frame_bgr, f"MAR: {mar:.3f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        # ── MAR bar gauge ───────────────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h = 10, 85, 180, 14
        fill_w = min(int(mar / 0.6 * bar_w), bar_w)
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      status_color, -1)
        cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (160, 160, 160), 1)
        # Threshold marker line
        thr_x = bar_x + int(self.lip_detector.mar_threshold / 0.6 * bar_w)
        cv2.line(frame_bgr, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2),
                 (255, 255, 0), 2)

        # ── Status text ─────────────────────────────────────────────────────
        cv2.putText(frame_bgr, status_text,
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # ── Mouth outline ────────────────────────────────────────────────────
        if mouth_pts:
            pts_arr = np.array(mouth_pts, dtype=np.int32)
            if self._mode == 'mediapipe':
                cv2.polylines(frame_bgr, [pts_arr], isClosed=True,
                              color=status_color, thickness=2)
                for pt in mouth_pts:
                    cv2.circle(frame_bgr, pt, 2, status_color, -1)
            else:
                # Haar mode: draw rectangle mouth ROI
                cv2.rectangle(frame_bgr, mouth_pts[0], mouth_pts[2],
                              status_color, 2)

        # ── Red border when actively talking ────────────────────────────────
        if is_talking:
            cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)

        # ── Mode badge (top-right corner) ────────────────────────────────────
        mode_label = f"LipDet: {self._mode.upper()}"
        cv2.putText(frame_bgr, mode_label, (w - 165, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

        return frame_bgr

    def close(self):
        """Release MediaPipe resources."""
        if self._face_landmarker:
            try:
                self._face_landmarker.close()
            except Exception:
                pass
        self._initialized = False


# ── Quick standalone test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    print("=" * 60)
    print("  ExamSentinelX AI — Talking / Lip Detector")
    print("=" * 60)

    detector = TalkingDetector(mar_threshold=MAR_THRESHOLD,
                               sustained_frames=SUSTAINED_FRAMES)
    ok = detector.initialize()

    if not ok:
        print("\n[!] All backends failed. Forcing Haar mode...")
        detector._try_init_haar()

    print(f"\n[OK] Active backend : {detector._mode.upper()}")
    print(f"[OK] MAR threshold  : {MAR_THRESHOLD}")
    print(f"[OK] Sustained frames: {SUSTAINED_FRAMES}")
    print("\n[*] Press Q to quit | Open/close mouth to test\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Cannot open webcam. Check camera connection.")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    fps_timer   = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[!] Camera read failed.")
            break

        frame_count += 1

        # ── Run detection on EVERY frame for real-time response ──────────────
        result = detector.process_frame(frame)

        # ── Draw detection overlay ───────────────────────────────────────────
        frame = detector.draw_visualization(frame, result)

        # ── Header bar ───────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
        cv2.putText(frame, "ExamSentinelX AI | Talking Detector",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        # FPS
        if frame_count % 15 == 0:
            elapsed = time.time() - fps_timer
            fps     = 15 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()
        else:
            fps = 0

        if fps > 0:
            cv2.putText(frame, f"FPS:{fps:.0f}", (w - 90, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        # ── Console output ───────────────────────────────────────────────────
        if result.get('is_talking'):
            print(f"\r[!!] TALKING DETECTED  "
                  f"MAR={result['mar']:.3f}  "
                  f"frames={result['open_duration_frames']:3d}    ",
                  end='', flush=True)
        elif result.get('mouth_open'):
            print(f"\r[ ] Mouth open  "
                  f"MAR={result['mar']:.3f}  "
                  f"frames={result['open_duration_frames']:3d}    ",
                  end='', flush=True)
        else:
            fd = result.get('face_detected', False)
            print(f"\r[OK] Monitoring  face={'YES' if fd else 'NO '}  "
                  f"MAR={result['mar']:.3f}           ",
                  end='', flush=True)

        _safe_imshow('ExamSentinelX - Talking Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\n\n[OK] Session ended.")
