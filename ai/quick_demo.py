"""
ai/quick_demo.py — Instant Exam Violation Demo (No Training Needed)
====================================================================
Uses YOLOv8n pretrained on COCO to detect cheating-related objects:
  - Extra persons (class 0)
  - Cell phones  (class 67)
  - Books        (class 73)
  - Laptops      (class 63)
  - Remotes      (class 65)

Works 100% offline with no dataset needed — just webcam + ultralytics.

Usage:
    python ai/quick_demo.py                 # live webcam
    python ai/quick_demo.py --image test.jpg # test with image
    python ai/quick_demo.py --video test.mp4 # test with video

CPU Performance: ~8-12 FPS on Core i5
"""

import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# ── Fix: capture the real cv2.imshow BEFORE ultralytics patches it ─────────────
# ultralytics monkey-patches cv2.imshow with a unicode_escape wrapper that
# crashes on Windows when the window name has special chars or when headless
# OpenCV is also present. We restore the original after YOLO loads.
_REAL_CV2_IMSHOW  = cv2.imshow
_REAL_CV2_WINDOW  = cv2.namedWindow

def _safe_imshow(winname: str, mat):
    """Drop-in for cv2.imshow that always uses the native GUI call."""
    # Strip any non-ASCII to be safe on all Windows locales
    safe_name = winname.encode('ascii', errors='replace').decode('ascii')
    _REAL_CV2_IMSHOW(safe_name, mat)

def _patch_ultralytics_imshow():
    """Restore real imshow after ultralytics patches it."""
    try:
        import ultralytics.utils.patches as _patches
        _patches.imshow = _safe_imshow
        # Also patch the cv2 module itself so any stray call works
        cv2.imshow = _safe_imshow
    except Exception:
        pass

# ── COCO classes relevant to exam cheating ─────────────────────────────────────
EXAM_CLASSES = {
    0:  ("person",      (0, 200, 255),   "EXTRA PERSON"),
    63: ("laptop",      (0, 0, 255),     "LAPTOP/DEVICE"),
    64: ("mouse",       (0, 100, 255),   "COMPUTER MOUSE"),
    65: ("remote",      (0, 0, 200),     "REMOTE/DEVICE"),
    67: ("cell phone",  (0, 0, 255),     "PHONE DETECTED"),
    73: ("book",        (255, 0, 0),     "UNAUTHORIZED BOOK"),
    76: ("scissors",    (255, 100, 0),   "SCISSORS"),
    84: ("book",        (255, 0, 100),   "NOTEBOOK"),
}

# Alert thresholds
CONFIDENCE_THRESHOLD = 0.40   # Min confidence to show detection
ALERT_THRESHOLD      = 0.50   # Min confidence to log as violation


def load_yolo():
    """Load YOLOv8n model (downloads ~6 MB on first run)."""
    try:
        from ultralytics import YOLO
        # Restore real imshow after ultralytics patches it
        _patch_ultralytics_imshow()

        # Prefer fine-tuned exam model if available
        custom_model = Path(__file__).parent / "models" / "yolov8_cheating_exam.pt"
        if custom_model.exists():
            print(f"[*] Loading fine-tuned exam model: {custom_model}")
            model = YOLO(str(custom_model))
            use_custom = True
        else:
            print("[*] Loading pretrained YOLOv8n (COCO)...")
            print("    (Fine-tuned model not found — run ai/train_model.py first)")
            model = YOLO("yolov8n.pt")
            use_custom = False
        
        print("[OK] Model loaded successfully!")
        return model, use_custom
    
    except ImportError:
        print("[!] ultralytics not installed. Run: pip install ultralytics")
        return None, False


def analyze_detections(results, use_custom=False):
    """
    Extract exam-relevant violations from YOLO results.
    Returns list of violation dicts.
    """
    violations = []
    
    for result in results:
        if result.boxes is None:
            continue
        
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf   = float(boxes.conf[i].item())
            xyxy   = boxes.xyxy[i].tolist()
            
            if use_custom:
                # Fine-tuned model: all classes are exam violations
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                class_names = result.names
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                violations.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "alert":      cls_name.upper(),
                    "confidence": conf,
                    "bbox":       xyxy,
                    "color":      (0, 0, 255),
                    "is_violation": conf >= ALERT_THRESHOLD,
                })
            else:
                # COCO model: filter to exam-relevant classes only
                if cls_id not in EXAM_CLASSES:
                    continue
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                
                cls_name, color, alert = EXAM_CLASSES[cls_id]
                violations.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "alert":      alert,
                    "confidence": conf,
                    "bbox":       xyxy,
                    "color":      color,
                    "is_violation": conf >= ALERT_THRESHOLD,
                })
    
    return violations


def draw_detections(frame, violations, fps=0):
    """Draw bounding boxes and alerts on frame."""
    h, w = frame.shape[:2]
    
    # ── Draw detections ────────────────────────────────────────────────────────
    for v in violations:
        x1, y1, x2, y2 = map(int, v["bbox"])
        color = v["color"]
        
        # Bounding box
        thickness = 3 if v["is_violation"] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label background
        label = f"{v['alert']} {v['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # ── HUD overlay ────────────────────────────────────────────────────────────
    # Status bar at top
    has_violation = any(v["is_violation"] for v in violations)
    status_color = (0, 0, 200) if has_violation else (0, 180, 0)
    status_text  = "!! VIOLATION DETECTED !!" if has_violation else "Monitoring..."
    
    cv2.rectangle(frame, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(frame, "ExamSentinelX AI", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (w // 2 - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Violation count bottom bar
    if violations:
        count_text = f"Detections: {len(violations)} | Violations: {sum(1 for v in violations if v['is_violation'])}"
        cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, count_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    
    return frame


def run_webcam(model, use_custom=False):
    """
    Run real-time detection from webcam.
    
    Uses a background thread for YOLO inference so the display loop
    always runs at full webcam FPS (~30), never freezing the window.
    YOLO input is scaled to 320px for ~4x faster CPU inference.
    """
    import threading

    print("\n[*] Starting webcam... Press Q to quit, S to save screenshot")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print("[!] Cannot open webcam. Check if camera is connected.")
        return

    # ── Shared state between display thread and inference thread ──────────────
    _lock            = threading.Lock()
    _latest_frame    = [None]       # Frame to run inference on
    _violations      = [[]]         # Last detected violations
    _inf_running     = [False]      # Is inference currently running?
    _stop            = [False]      # Signal to stop threads

    # Save directory
    save_dir = Path(__file__).parent.parent / "captures" / "demo"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Background inference thread ────────────────────────────────────────────
    def inference_worker():
        while not _stop[0]:
            with _lock:
                frame = _latest_frame[0]
                if frame is None or _inf_running[0]:
                    continue
                _inf_running[0] = True

            try:
                # imgsz=320: 4x faster than 640 on CPU, still good detection
                results = model(frame, verbose=False,
                                conf=CONFIDENCE_THRESHOLD, imgsz=320)
                viols = analyze_detections(results, use_custom)
            except Exception:
                viols = []

            with _lock:
                _violations[0] = viols
                _inf_running[0] = False

            time.sleep(0.01)  # Yield CPU briefly

    inf_thread = threading.Thread(target=inference_worker, daemon=True)
    inf_thread.start()

    print("[OK] Webcam started. Monitoring exam environment...")
    print("     YOLO running in background thread (imgsz=320 for CPU speed)\n")

    fps_timer   = time.time()
    frame_count = 0
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Feed the latest frame to the inference thread (non-blocking)
        with _lock:
            _latest_frame[0] = frame.copy()
            violations = list(_violations[0])
            running    = _inf_running[0]

        # FPS
        if frame_count % 15 == 0:
            elapsed = time.time() - fps_timer
            fps     = 15 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        # Draw detections on current frame
        display = draw_detections(frame, violations, fps)

        # "Analyzing..." badge when inference is in progress
        if running:
            h, w = display.shape[:2]
            cv2.putText(display, "Analyzing...", (w - 160, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        _safe_imshow("ExamSentinelX AI - Live Monitoring", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(path), display)
            print(f"[OK] Screenshot saved: {path}")

    _stop[0] = True
    inf_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("\n[OK] Monitoring session ended.")



def run_image(model, image_path: str, use_custom=False):
    """Run detection on a single image."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[!] Cannot load image: {image_path}")
        return
    
    print(f"[*] Analyzing: {image_path}")
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    violations = analyze_detections(results, use_custom)
    
    frame = draw_detections(frame, violations)
    
    # Print violations
    if violations:
        print(f"\n[!] {len(violations)} detection(s) found:")
        for v in violations:
            icon = "!!" if v["is_violation"] else "--"
            print(f"  {icon} {v['alert']} (confidence: {v['confidence']:.1%})")
    else:
        print("[OK] No exam violations detected.")
    
    cv2.imshow("ExamSentinelX AI - Analysis Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model, video_path: str, use_custom=False):
    """Run detection on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Cannot open video: {video_path}")
        return
    
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30
    total_violations = 0
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    last_violations = []
    
    print(f"[*] Analyzing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 2 == 0:  # Process every other frame on CPU
            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
            last_violations = analyze_detections(results, use_custom)
            total_violations += sum(1 for v in last_violations if v["is_violation"])
        violations = last_violations  # Persist last result
        
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_timer
            fps = 10 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()
        
        frame = draw_detections(frame, violations, fps)
        cv2.imshow("ExamSentinelX AI - Video Analysis", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] Video analysis complete.")
    print(f"     Total frames: {frame_count}")
    print(f"     Total violations: {total_violations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExamSentinelX AI Quick Demo")
    parser.add_argument("--image", type=str, default=None, help="Test image path")
    parser.add_argument("--video", type=str, default=None, help="Test video path")
    args = parser.parse_args()
    
    model, use_custom = load_yolo()
    if model is None:
        exit(1)
    
    if args.image:
        run_image(model, args.image, use_custom)
    elif args.video:
        run_video(model, args.video, use_custom)
    else:
        run_webcam(model, use_custom)
