"""
test_all_detections.py — Full AI Detection Pipeline Test (Terminal + OpenCV Window)
====================================================================================
Tests ALL detection modules on live webcam simultaneously:
  1. Face Detection   (Haar Cascade)
  2. Head Pose        (Centroid-shift)
  3. Object Detection (YOLOv8 — phone, book, laptop, person)
  4. Talking/Lip      (MediaPipe or Haar fallback)

Usage:
    python test_all_detections.py

Controls:
    Q = Quit
    S = Save screenshot
"""

import cv2
import time
import sys
import os
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Fix ultralytics cv2.imshow monkey-patching ────────────────────────────────
_REAL_IMSHOW = cv2.imshow

def _safe_imshow(winname: str, mat):
    safe = winname.encode('ascii', errors='replace').decode('ascii')
    _REAL_IMSHOW(safe, mat)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import AI modules ─────────────────────────────────────────────────────────
from ai.face_detector import FaceDetector
from ai.head_pose_estimator import HeadPoseEstimator
from ai.object_detector import ObjectDetector
from ai.lip_detector import TalkingDetector

# Patch ultralytics imshow after import
try:
    import ultralytics.utils.patches as _p
    _p.imshow = _safe_imshow
    cv2.imshow = _safe_imshow
except Exception:
    pass


# ── Colour constants ──────────────────────────────────────────────────────────
GREEN   = (0, 220, 100)
RED     = (0, 0, 255)
CYAN    = (255, 220, 0)
YELLOW  = (0, 255, 255)
WHITE   = (255, 255, 255)
GREY    = (160, 160, 160)
ORANGE  = (0, 165, 255)
BG_DARK = (20, 20, 25)


def draw_hud(frame, face_result, pose_result, obj_result, talk_result, fps, trust_score):
    """Draw a comprehensive HUD overlay showing all detector statuses."""
    h, w = frame.shape[:2]

    # ── Top status bar ──────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), BG_DARK, -1)
    cv2.putText(frame, "ExamSentinelX AI", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, CYAN, 2)
    cv2.putText(frame, "ALL DETECTORS TEST", (280, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)

    # ── Right-side panel (status of each detector) ──────────────────────────
    panel_x = w - 320
    panel_y = 60
    panel_w = 310
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
    open_frames = talk_result.get('open_duration_frames', 0) if talk_result else 0
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
    violations = []
    if not face_ok:
        violations.append("NO FACE")
    if phone:
        violations.append("PHONE")
    if persons > 1:
        violations.append("MULTI-PERSON")
    if pose_violation:
        violations.append("LOOKING AWAY")
    if is_talking:
        violations.append("TALKING")
    if obj_result and obj_result.suspicious_objects:
        violations.append("SUSPICIOUS OBJ")

    # Draw Trust Score
    y += 5
    ts_color = GREEN if trust_score > 80 else (ORANGE if trust_score > 50 else RED)
    cv2.putText(frame, f"TRUST SCORE: {trust_score:.1f}%", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ts_color, 2)
    y += 25

    if violations:
        overall = " | ".join(violations)
        cv2.putText(frame, "ALERT:", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        # Wrap long text
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
    if violations:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 4)

    return frame


def main():
    print("=" * 65)
    print("  ExamSentinelX AI — Full Detection Pipeline Test")
    print("=" * 65)
    print()

    # ── Initialize all detectors ────────────────────────────────────────────
    print("[1/4] Initializing Face Detector (Haar Cascade)...")
    face_detector = FaceDetector(min_detection_confidence=0.7)
    print("      [OK] Face detector ready")

    print("[2/4] Initializing Head Pose Estimator...")
    head_pose = HeadPoseEstimator(yaw_limit=30, pitch_limit=20)
    print("      [OK] Head pose estimator ready")

    print("[3/4] Initializing YOLOv8 Object Detector...")
    obj_detector = ObjectDetector(confidence_threshold=0.45)
    print("      [OK] Object detector ready")

    print("[4/4] Initializing Talking Detector...")
    talk_detector = TalkingDetector(mar_threshold=0.28, sustained_frames=5)
    talk_detector.initialize()
    print(f"      [OK] Talking detector ready (mode: {talk_detector._mode})")

    print()
    print("[*] Opening webcam...")

    # ── Open webcam ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[!] ERROR: Cannot open webcam!")
        return

    print("[OK] Webcam opened successfully.")
    print()
    print("-" * 65)
    print("  Controls:  Q = Quit  |  S = Save screenshot")
    print("  Test:      Show your face, then try phone, look away, talk...")
    print("-" * 65)
    print()

    # ── Shared state for threaded YOLO inference ────────────────────────────
    _lock = threading.Lock()
    _latest_frame = [None]
    _obj_result = [None]
    _yolo_running = [False]
    _stop = [False]

    def yolo_worker():
        """Run YOLO in background thread to keep display smooth."""
        while not _stop[0]:
            with _lock:
                frame = _latest_frame[0]
                if frame is None or _yolo_running[0]:
                    time.sleep(0.01)
                    continue
                _yolo_running[0] = True

            try:
                result = obj_detector.detect(frame, draw_annotations=True)
            except Exception:
                result = None

            with _lock:
                _obj_result[0] = result
                _yolo_running[0] = False

            time.sleep(0.01)

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
    yolo_thread.start()

    save_dir = PROJECT_ROOT / "captures" / "test_output"
    save_dir.mkdir(parents=True, exist_ok=True)

    fps_timer = time.time()
    frame_count = 0
    fps = 0.0

    # Last printed status (to avoid terminal spam — only print on change)
    last_printed = ""
    
    trust_score = 100.0
    cooldowns = {}
    
    def apply_penalty(v_type, penalty):
        nonlocal trust_score
        now = time.time()
        if now - cooldowns.get(v_type, 0) > 6.0:
            trust_score = max(0.0, trust_score - penalty)
            cooldowns[v_type] = now

    # Try to import ctypes for window focus detection (mocking tab switch)
    has_ctypes = False
    try:
        import ctypes
        has_ctypes = True
    except:
        pass

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ── 1. Face Detection ───────────────────────────────────────────────
        face_result = face_detector.detect(frame, draw_annotations=True)
        face_frame = face_result.annotated_frame if face_result.annotated_frame is not None else frame

        # ── 2. Head Pose ────────────────────────────────────────────────────
        pose_result = head_pose.estimate(face_frame, draw_annotations=True)
        pose_frame = pose_result.annotated_frame if pose_result.annotated_frame is not None else face_frame

        # ── 3. YOLO Objects (threaded — non-blocking) ───────────────────────
        with _lock:
            _latest_frame[0] = frame.copy()
            obj_result = _obj_result[0]

        # Use YOLO-annotated frame if available
        display_frame = pose_frame

        # ── 4. Talking Detection ────────────────────────────────────────────
        try:
            talk_result = talk_detector.process_frame(frame)
        except Exception:
            talk_result = {}

        # ── 5. App Switch Detection (Mocking Tab Switch) ────────────────────
        app_switched = False
        if has_ctypes:
            hwnd_active = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd_active)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd_active, buf, length + 1)
            active_title = buf.value
            if active_title and "ExamSentinelX" not in active_title and "python" not in active_title.lower():
                app_switched = True

        # ── Apply Penalties ─────────────────────────────────────────────────
        face_count = face_result.face_count if face_result else 0
        direction = pose_result.direction if pose_result else "N/A"
        pose_v = pose_result.violation_type if pose_result else None
        phone = obj_result.phone_detected if obj_result else False
        persons = obj_result.person_count if obj_result else 0
        talking = talk_result.get('is_talking', False) if talk_result else False

        if face_count == 0: apply_penalty('face_absent', 5.0)
        if phone: apply_penalty('phone', 20.0)
        if persons > 1: apply_penalty('multi_person', 15.0)
        if pose_v: apply_penalty('looking_away', 3.0)
        if talking: apply_penalty('talking', 12.0)
        if app_switched: apply_penalty('app_switch', 10.0)

        # ── FPS calculation ─────────────────────────────────────────────────
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # ── Draw HUD ────────────────────────────────────────────────────────
        display = draw_hud(display_frame, face_result, pose_result,
                           obj_result, talk_result, fps, trust_score)

        _safe_imshow("ExamSentinelX AI - Full Test", display)

        # ── Terminal output (only on status change) ─────────────────────────
        face_count = face_result.face_count if face_result else 0
        direction = pose_result.direction if pose_result else "N/A"
        pose_v = pose_result.violation_type if pose_result else None
        phone = obj_result.phone_detected if obj_result else False
        persons = obj_result.person_count if obj_result else 0
        talking = talk_result.get('is_talking', False) if talk_result else False
        mar = talk_result.get('mar', 0.0) if talk_result else 0.0

        status = (f"Face:{face_count} | Pose:{direction}"
                  f"{' ⚠AWAY' if pose_v else ''}"
                  f" | Phone:{'YES!' if phone else 'no'}"
                  f" | Persons:{persons}"
                  f" | Talk:{'YES!' if talking else 'no'}"
                  f" MAR:{mar:.2f}")

        if status != last_printed:
            ts = datetime.now().strftime("%H:%M:%S")
            # Color-code the terminal output
            alerts = []
            if face_count == 0:
                alerts.append("NO_FACE")
            if phone:
                alerts.append("PHONE")
            if persons > 1:
                alerts.append("MULTI-PERSON")
            if pose_v:
                alerts.append("LOOKING_AWAY")
            if talking:
                alerts.append("TALKING")
            if app_switched:
                alerts.append("APP_SWITCHED")

            if alerts:
                print(f"[{ts}] [!] {status} | TS:{trust_score:.1f}% <-- ALERT: {', '.join(alerts)}")
            else:
                print(f"[{ts}] [OK] {status} | TS:{trust_score:.1f}%")
            last_printed = status

        # ── Key handling ────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"test_{ts}.jpg"
            cv2.imwrite(str(path), display)
            print(f"[SAVE] Screenshot → {path}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    _stop[0] = True
    yolo_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    face_detector.close()
    head_pose.close()
    print("\n[OK] Test session ended.")


if __name__ == "__main__":
    main()
