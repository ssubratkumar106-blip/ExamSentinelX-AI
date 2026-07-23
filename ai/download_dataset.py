"""
ai/download_dataset.py — Dataset Downloader for ExamSentinelX AI
=============================================================
Downloads the "Cheating Online Exam" COCO dataset from Roboflow Universe
and prepares it for YOLOv8 fine-tuning.

Dataset Source:
    https://universe.roboflow.com/cheating-online-exam/cheating-online-exam
    Classes: cheating, phone, book, paper, normal

Usage:
    python ai/download_dataset.py

What it does:
    1. Downloads dataset via Roboflow Python SDK
    2. Converts COCO JSON to YOLO format (if needed)
    3. Creates proper data.yaml
    4. Validates all splits (train/val/test)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (useful if run directly)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# ── Project root ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "ai" / "datasets"
MODELS_DIR  = BASE_DIR / "ai" / "models"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_via_roboflow_sdk():
    """
    Method 1: Download using the official Roboflow Python SDK.
    Requires: pip install roboflow
    Get your free API key at: https://app.roboflow.com/settings/api
    """
    try:
        from roboflow import Roboflow
        print("[*] Roboflow SDK found.")
    except ImportError:
        print("[!] Roboflow SDK not installed.")
        print("    Run: pip install roboflow")
        return False

    # ── You need a free Roboflow API key ──────────────────────────────────────
    # Get it free at: https://app.roboflow.com/settings/api (no credit card)
    api_key = os.getenv("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("\n" + "="*60)
        print("  ROBOFLOW API KEY REQUIRED")
        print("="*60)
        print("  1. Go to: https://app.roboflow.com/settings/api")
        print("  2. Copy your free API key")
        print("  3. Set it: set ROBOFLOW_API_KEY=your_key_here")
        print("  4. Re-run this script")
        print("="*60)
        print("\n  OR use Method 2 (manual ZIP download) below.\n")
        return False

    rf = Roboflow(api_key=api_key)

    # ── Cheating Online Exam Dataset (COCO format) ────────────────────────────
    # Dataset: https://universe.roboflow.com/cheating-online-exam/cheating-online-exam
    try:
        project = rf.workspace("cheating-online-exam").project("cheating-online-exam")
        dataset = project.version(1).download("yolov8", location=str(DATASET_DIR / "cheating_exam"))
        print(f"[OK] Dataset downloaded to: {DATASET_DIR / 'cheating_exam'}")
        return True
    except Exception as e:
        print(f"[!] Failed to download via SDK: {e}")
        print("    Try the workspace/project name from Roboflow URL.")
        return False


def download_coco_format_manual():
    """
    Method 2: Guide for manual download + COCO→YOLO conversion.
    User downloads the ZIP from Roboflow, extracts it, then runs this.
    """
    print("\n" + "="*60)
    print("  MANUAL DOWNLOAD GUIDE")
    print("="*60)
    print("""
  Step 1: Visit this URL in your browser:
  https://universe.roboflow.com/cheating-online-exam/cheating-online-exam

  Step 2: Click "Download Dataset"

  Step 3: Select format: "COCO" or "YOLOv8"

  Step 4: Download and extract the ZIP to:
  ExamSentinelXAI/ai/datasets/cheating_exam/

  The folder should look like:
  datasets/cheating_exam/
    train/
      images/   ← .jpg files
      labels/   ← .txt files (YOLO format)
    valid/
      images/
      labels/
    test/
      images/
      labels/
    data.yaml   ← class definitions

  Step 5: Run:  python ai/train_model.py
""")


def convert_coco_to_yolo(coco_json_path: str, output_dir: str, split: str = "train"):
    """
    Convert COCO JSON annotations to YOLO TXT format.
    
    YOLO format per line:
        <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        output_dir: Where to write YOLO .txt files
        split: 'train', 'valid', or 'test'
    """
    print(f"[*] Converting COCO -> YOLO for split: {split}")

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category ID → class index mapping
    categories = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
    cat_names   = [cat['name'] for cat in coco['categories']]

    # Build image ID → filename + dimensions mapping
    images = {img['id']: img for img in coco['images']}

    # Group annotations by image ID
    ann_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        ann_by_image[img_id].append(ann)

    labels_dir = Path(output_dir) / split / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, img_info in images.items():
        W = img_info['width']
        H = img_info['height']
        filename = Path(img_info['file_name']).stem

        lines = []
        for ann in ann_by_image.get(img_id, []):
            cls_idx = categories[ann['category_id']]
            x, y, w, h = ann['bbox']  # COCO: x_min, y_min, width, height

            # Convert to YOLO normalized center format
            xc = (x + w / 2) / W
            yc = (y + h / 2) / H
            wn = w / W
            hn = h / H

            # Clamp to [0, 1]
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            wn = max(0, min(1, wn))
            hn = max(0, min(1, hn))

            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # Write label file
        label_file = labels_dir / f"{filename}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(lines))
        converted += 1

    print(f"    [OK] Converted {converted} images for {split}")
    return cat_names


def create_data_yaml(dataset_dir: str, class_names: list):
    """Create the data.yaml file required by YOLOv8 training."""
    yaml_content = f"""# ExamSentinelX AI — Cheating Detection Dataset
# Auto-generated by download_dataset.py

path: {dataset_dir}
train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = Path(dataset_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"[OK] data.yaml created: {yaml_path}")
    return str(yaml_path)


def validate_dataset(dataset_dir: str):
    """Check that the dataset structure is correct before training."""
    dataset_path = Path(dataset_dir)
    required = [
        "train/images", "train/labels",
        "valid/images", "valid/labels",
    ]
    all_ok = True
    for rel_path in required:
        full = dataset_path / rel_path
        if not full.exists():
            print(f"[MISS] {full}")
            all_ok = False
        else:
            count = len(list(full.glob("*")))
            print(f"[OK]   {full}  ({count} files)")

    # Check data.yaml
    yaml = dataset_path / "data.yaml"
    if yaml.exists():
        print(f"[OK]   data.yaml found")
    else:
        print(f"[MISS] data.yaml not found")
        all_ok = False

    return all_ok


def process_coco_zip(zip_path: str):
    """
    Process a downloaded COCO ZIP from Roboflow.
    Handles conversion if annotations are in COCO JSON format.
    """
    import zipfile

    extract_dir = DATASET_DIR / "cheating_exam_raw"
    output_dir  = DATASET_DIR / "cheating_exam"

    print(f"[*] Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(str(extract_dir))
    print(f"[OK] Extracted to {extract_dir}")

    # Check if COCO JSON exists
    coco_files = list(extract_dir.rglob("_annotations.coco.json"))

    if coco_files:
        print(f"[*] COCO format detected. Converting to YOLO...")
        class_names = None
        for coco_file in coco_files:
            # Determine split from parent folder name
            split_name = coco_file.parent.name
            if split_name not in ['train', 'valid', 'test']:
                split_name = 'train'

            # Copy images
            img_src = coco_file.parent
            img_dst = output_dir / split_name / "images"
            img_dst.mkdir(parents=True, exist_ok=True)
            for img in img_src.glob("*.jpg"):
                shutil.copy(str(img), str(img_dst / img.name))
            for img in img_src.glob("*.png"):
                shutil.copy(str(img), str(img_dst / img.name))

            names = convert_coco_to_yolo(str(coco_file), str(output_dir), split_name)
            if class_names is None:
                class_names = names

        if class_names:
            create_data_yaml(str(output_dir), class_names)
    else:
        # Already YOLO format — just copy
        print(f"[*] YOLO format detected. Copying...")
        if extract_dir != output_dir:
            shutil.copytree(str(extract_dir), str(output_dir), dirs_exist_ok=True)

    return str(output_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("  ExamSentinelX AI — Dataset Downloader")
    print("=" * 60)

    # Try SDK download first
    success = download_via_roboflow_sdk()

    if not success:
        # Show manual instructions
        download_coco_format_manual()

        # Check if user already extracted manually
        manual_path = DATASET_DIR / "cheating_exam"
        if manual_path.exists():
            print(f"\n[*] Found existing dataset at: {manual_path}")
            print("[*] Validating structure...")
            ok = validate_dataset(str(manual_path))
            if ok:
                print("\n[OK] Dataset is ready. Run: python ai/train_model.py")
            else:
                print("\n[!] Dataset structure incomplete. Please check the guide above.")
        else:
            print(f"\n[?] No dataset found at {manual_path}")
            print("    Please download and extract to that folder first.")

    sys.exit(0 if success else 1)
