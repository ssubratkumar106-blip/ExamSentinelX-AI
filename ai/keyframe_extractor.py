"""
ai/keyframe_extractor.py
========================
Motion-Based Keyframe Extraction
Paper: Section III-B-1

Algorithm from paper:
  T = mean(absdiff) + std(absdiff)   [threshold formula]
  threshold = 340,000                 [paper's optimal value]
  skip_factor = 3                     [frame skip]

  For each pair of successive frames:
    absdiff_f = abs(current_frame - previous_frame)
    avgdiff   = mean(absdiff_f)
    if avgdiff > T: → keyframe
    else:           → skip
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple


class KeyframeExtractor:
    """
    Extracts keyframes from video based on motion (pixel difference).
    Implements the exact algorithm from the IEEE paper.
    """

    def __init__(self, threshold: float = 340000, skip_factor: int = 3):
        """
        Args:
            threshold:    Motion threshold T (paper uses 340,000).
                          Higher → fewer keyframes extracted.
            skip_factor:  Sample every N-th frame before comparison.
                          Paper uses skip_factor=3.
        """
        self.threshold = threshold
        self.skip_factor = skip_factor

    def extract_from_video(
        self,
        video_path: str,
        output_dir: str = None,
        resize: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """
        Extract keyframes from a video file.

        Args:
            video_path:  Path to input video (.mp4, .avi, etc.)
            output_dir:  If given, save keyframes as JPEG images here.
            resize:      Resize each frame to this size (paper: 224×224×3).

        Returns:
            List of keyframe images as numpy arrays (BGR, uint8).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        keyframes = []
        prev_frame_gray = None
        frame_idx = 0
        keyframe_idx = 0

        # --- Compute adaptive threshold from first pass (optional) ---
        # Paper uses fixed T=340000, but we also support adaptive mode
        T = self.threshold

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply skip factor — only process every N-th frame
            if frame_idx % self.skip_factor != 0:
                frame_idx += 1
                continue

            # Resize to paper's standard size (224×224)
            frame_resized = cv2.resize(frame, resize)

            # Apply Gaussian filter (paper Section III-B: noise removal)
            frame_filtered = cv2.GaussianBlur(frame_resized, (5, 5), 0)

            # Apply histogram equalization for brightness normalization
            frame_yuv = cv2.cvtColor(frame_filtered, cv2.COLOR_BGR2YUV)
            frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
            frame_normalized = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

            # Convert to grayscale for diff computation
            gray = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2GRAY).astype(np.float32)

            if prev_frame_gray is None:
                # Always keep the first frame
                prev_frame_gray = gray
                keyframes.append(frame_normalized)
                keyframe_idx += 1
                frame_idx += 1
                continue

            # Paper Equation (1): absdiff_f = abs(current - previous)
            absdiff = np.abs(gray - prev_frame_gray)

            # Paper Equation (2): avgdiff = mean(absdiff_f)
            avgdiff = np.mean(absdiff)

            # Paper Equation (3): if avgdiff > T → keyframe
            if avgdiff > T:
                keyframes.append(frame_normalized)
                keyframe_idx += 1

                # Save to disk if output_dir specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    out_path = os.path.join(output_dir, f"keyframe_{keyframe_idx:05d}.jpg")
                    cv2.imwrite(out_path, frame_normalized)

            # Update previous frame
            prev_frame_gray = gray
            frame_idx += 1

        cap.release()
        print(f"[KeyframeExtractor] {video_path}: {frame_idx} frames → {len(keyframes)} keyframes")
        return keyframes

    def extract_from_dataset(
        self,
        dataset_dir: str,
        output_dir: str,
        class_names: List[str] = None,
        resize: Tuple[int, int] = (224, 224)
    ) -> dict:
        """
        Process an entire dataset directory with class subdirectories.

        Expected structure:
            dataset_dir/
                external_device/  ← video files
                head_movement/
                multiple_persons/
                talking_to_others/
                normal/

        Output structure:
            output_dir/
                external_device/  ← keyframe JPEGs
                head_movement/
                ...

        Returns:
            Dict mapping class_name → number of keyframes extracted.
        """
        if class_names is None:
            class_names = [
                'external_device',
                'head_movement',
                'multiple_persons',
                'talking_to_others',
                'normal'
            ]

        stats = {}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

        for class_name in class_names:
            class_dir = Path(dataset_dir) / class_name
            if not class_dir.exists():
                print(f"[KeyframeExtractor] Skipping missing class dir: {class_dir}")
                continue

            out_class_dir = Path(output_dir) / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            total_keyframes = 0
            video_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in video_extensions
            ]

            print(f"[KeyframeExtractor] Class '{class_name}': {len(video_files)} videos")

            for video_file in video_files:
                video_out = out_class_dir / video_file.stem
                try:
                    kf = self.extract_from_video(
                        str(video_file),
                        output_dir=str(video_out),
                        resize=resize
                    )
                    total_keyframes += len(kf)
                except Exception as e:
                    print(f"[KeyframeExtractor] Error on {video_file.name}: {e}")

            stats[class_name] = total_keyframes
            print(f"[KeyframeExtractor] '{class_name}': {total_keyframes} total keyframes")

        return stats

    def compute_adaptive_threshold(self, video_path: str) -> float:
        """
        Compute adaptive threshold T = mean(absdiff) + std(absdiff)
        from the first 100 frames of a video.
        Paper formula from Section III-B-1.
        """
        cap = cv2.VideoCapture(video_path)
        diffs = []
        prev_gray = None
        count = 0

        while count < 100:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(
                cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2GRAY
            ).astype(np.float32)

            if prev_gray is not None:
                absdiff = np.abs(gray - prev_gray)
                diffs.append(np.mean(absdiff))

            prev_gray = gray
            count += 1

        cap.release()

        if not diffs:
            return self.threshold

        T = np.mean(diffs) + np.std(diffs)
        print(f"[KeyframeExtractor] Adaptive threshold T = {T:.2f}")
        return T


# ── CLI Usage ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Motion-based keyframe extraction')
    parser.add_argument('--input', required=True, help='Input video file or dataset directory')
    parser.add_argument('--output', required=True, help='Output directory for keyframes')
    parser.add_argument('--threshold', type=float, default=340000,
                        help='Motion threshold T (paper default: 340000)')
    parser.add_argument('--skip', type=int, default=3,
                        help='Frame skip factor (paper default: 3)')
    parser.add_argument('--dataset', action='store_true',
                        help='Process entire class-structured dataset directory')
    args = parser.parse_args()

    extractor = KeyframeExtractor(threshold=args.threshold, skip_factor=args.skip)

    if args.dataset:
        stats = extractor.extract_from_dataset(args.input, args.output)
        print("\n=== Extraction Summary ===")
        total = 0
        for cls, count in stats.items():
            print(f"  {cls:25s}: {count:5d} keyframes")
            total += count
        print(f"  {'TOTAL':25s}: {total:5d} keyframes")
    else:
        keyframes = extractor.extract_from_video(args.input, args.output)
        print(f"Extracted {len(keyframes)} keyframes → {args.output}")
