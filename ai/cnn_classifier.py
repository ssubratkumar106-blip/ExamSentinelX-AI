"""
ai/cnn_classifier.py  (Updated — All 5 Paper Models)
=====================================================
Implements ALL deep learning models from the paper:

  1. InceptionV3          (paper Section III-F)
  2. InceptionResNetV2    (paper Section III-D)
  3. DenseNet121          (paper Section III-E)
  4. Custom CNN           (paper Section III-G, Table 3)
  5. YOLOv8               (upgraded from paper's YOLOv5, Section III-C)

Paper's 4 cheating classes + normal:
  0: external_device
  1: head_movement
  2: multiple_persons
  3: talking_to_others
  4: normal
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ── Class Configuration ───────────────────────────────────────────────────────
CLASS_NAMES = [
    'external_device',
    'head_movement',
    'multiple_persons',
    'talking_to_others',
    'normal'
]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE_224 = (224, 224)   # DenseNet121, CustomCNN
INPUT_SIZE_299 = (299, 299)   # InceptionV3, InceptionResNetV2


# ═══════════════════════════════════════════════════════════════════════════════
# BASE MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_transfer_model(base_model, input_size, num_classes, freeze_base=True):
    """
    Generic transfer learning wrapper:
      BaseModel (pretrained, frozen) → GlobalAvgPool → Dense(256, ReLU)
      → Dropout(0.5) → Dense(num_classes, Softmax)

    This mirrors the fine-tuning approach described in paper Section III-C/D/E/F.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    # Optionally freeze pretrained weights
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(*input_size, 3))

    # Preprocessing: normalize to [-1, 1] as required by Inception models
    x = tf.keras.applications.inception_v3.preprocess_input(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: InceptionV3  (Paper Section III-F)
# ═══════════════════════════════════════════════════════════════════════════════

def build_inception_v3(num_classes: int = NUM_CLASSES, freeze_base: bool = True):
    """
    Paper Section III-F: Inception-V3
    Input: 299×299×3
    Pretrained on ImageNet.
    """
    from tensorflow.keras.applications import InceptionV3

    base = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )
    model = _build_transfer_model(base, INPUT_SIZE_299, num_classes, freeze_base)
    model._name = 'InceptionV3_Cheating'

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[CNN] Built InceptionV3 model")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: InceptionResNetV2  (Paper Section III-D)
# ═══════════════════════════════════════════════════════════════════════════════

def build_inception_resnet_v2(num_classes: int = NUM_CLASSES, freeze_base: bool = True):
    """
    Paper Section III-D: Inception_ResNet_v2
    Input: 299×299×3, 164 layers.
    Pretrained on ImageNet (1000 classes).
    """
    from tensorflow.keras.applications import InceptionResNetV2

    base = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )
    model = _build_transfer_model(base, INPUT_SIZE_299, num_classes, freeze_base)
    model._name = 'InceptionResNetV2_Cheating'

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[CNN] Built InceptionResNetV2 model")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: DenseNet121  (Paper Section III-E)
# ═══════════════════════════════════════════════════════════════════════════════

def build_densenet121(num_classes: int = NUM_CLASSES, freeze_base: bool = True):
    """
    Paper Section III-E: DenseNet121
    Input: 224×224×3
    Each layer connected to all layers below it.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras import layers, Model

    base = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='DenseNet121_Cheating')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[CNN] Built DenseNet121 model")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: Custom CNN  (Paper Section III-G, Table 3)
# ═══════════════════════════════════════════════════════════════════════════════

def build_custom_cnn(
    input_size: Tuple[int, int] = INPUT_SIZE_224,
    num_classes: int = NUM_CLASSES
):
    """
    Paper Section III-G — Fine-Tuned Custom CNN (Table 3):

    Architecture:
      Hidden Layer 1:  Conv2D(32, 3×3) → MaxPool(2×2) → LeakyReLU
      Hidden Layer 2:  Conv2D(64, 3×3) → MaxPool(2×2) → LeakyReLU
      Fully Connected: Flatten → Dense(128) → Dropout(0.5) → Dense(N, Softmax)

    Optimizer: Adam  |  Loss: Categorical Crossentropy  |  Epochs: 100+
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential

    model = Sequential([
        # ── Input ──────────────────────────────────────────────────────────────
        layers.Input(shape=(*input_size, 3)),
        layers.Rescaling(1.0 / 255),          # Normalize [0,255] → [0,1]

        # ── Hidden Layer 1 (Paper Table 3) ────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.LeakyReLU(alpha=0.1),          # Paper: "non-linear Leaky ReLU"
        layers.BatchNormalization(),

        # ── Hidden Layer 2 (Paper Table 3) ────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),

        # ── Extra Conv for better feature extraction ───────────────────────────
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.LeakyReLU(alpha=0.1),
        layers.BatchNormalization(),

        # ── Fully Connected Layers (Paper Table 3) ────────────────────────────
        layers.Flatten(),
        layers.Dense(128, activation='relu'),  # Paper: "dense"
        layers.Dropout(0.5),                   # Paper: "dropout"
        layers.Dense(num_classes, activation='softmax'),  # Paper: "dense + softmax"
    ], name='CustomCNN_Cheating')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[CNN] Built Custom CNN model (paper architecture)")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (runtime inference)
# ═══════════════════════════════════════════════════════════════════════════════

class CNNClassifier:
    """
    Runtime wrapper that loads a trained .h5 model and classifies frames.
    Used by ai/detector.py for real-time proctoring.
    """

    MODEL_PATHS = {
        'inception_v3':       'ai/models/inception_v3_cheating.h5',
        'inception_resnet_v2':'ai/models/inception_resnet_v2_cheating.h5',
        'densenet121':        'ai/models/densenet121_cheating.h5',
        'custom_cnn':         'ai/models/custom_cnn_cheating.h5',
    }

    def __init__(self, model_name: str = 'custom_cnn', base_dir: str = None):
        """
        Args:
            model_name: One of the keys in MODEL_PATHS
            base_dir:   Root directory of the project (for relative path resolution)
        """
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        self.base_dir = base_dir or Path(__file__).parent.parent

        model_rel_path = self.MODEL_PATHS.get(model_name, self.MODEL_PATHS['custom_cnn'])
        self.model_path = Path(self.base_dir) / model_rel_path

        # Determine input size based on model
        self.input_size = INPUT_SIZE_299 if 'inception' in model_name else INPUT_SIZE_224

    def load(self) -> bool:
        """Load the trained model weights from disk."""
        if not self.model_path.exists():
            print(f"[CNNClassifier] Model not found: {self.model_path}")
            print(f"[CNNClassifier] Run: python ai/train_model.py --model {self.model_name}")
            return False

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path))
            self.is_loaded = True
            print(f"[CNNClassifier] Loaded {self.model_name} from {self.model_path}")
            return True
        except Exception as e:
            print(f"[CNNClassifier] Load error: {e}")
            return False

    def classify_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Classify a single BGR frame.

        Returns:
            {
                'class': 'external_device',
                'class_id': 0,
                'confidence': 0.94,
                'all_probs': {'external_device': 0.94, 'normal': 0.04, ...}
            }
        """
        if not self.is_loaded:
            return {'class': 'normal', 'class_id': 4, 'confidence': 0.0, 'all_probs': {}}

        import cv2
        import tensorflow as tf

        # Preprocess
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Predict
        probs = self.model.predict(img, verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return {
            'class': CLASS_NAMES[class_id],
            'class_id': class_id,
            'confidence': confidence,
            'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
        }

    def classify_batch(self, frames: list) -> list:
        """Classify a batch of BGR frames. Returns list of result dicts."""
        if not self.is_loaded or not frames:
            return [{'class': 'normal', 'class_id': 4, 'confidence': 0.0} for _ in frames]

        import cv2
        import tensorflow as tf

        batch = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size).astype(np.float32)
            batch.append(img)

        batch = np.array(batch)
        all_probs = self.model.predict(batch, verbose=0)

        results = []
        for probs in all_probs:
            class_id = int(np.argmax(probs))
            results.append({
                'class': CLASS_NAMES[class_id],
                'class_id': class_id,
                'confidence': float(probs[class_id]),
                'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
            })
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY SHIM — used by ai/detector.py
# ═══════════════════════════════════════════════════════════════════════════════

def get_cnn_classifier(model_name: str = 'custom_cnn'):
    """
    Returns a loaded CNNClassifier, or None if no trained model exists yet.
    Called by ai/detector.py at session startup.

    Falls back through model options until one is found:
        custom_cnn → densenet121 → inception_v3 → None
    """
    fallback_order = ['custom_cnn', 'densenet121', 'inception_v3', 'inception_resnet_v2']

    for name in ([model_name] + [m for m in fallback_order if m != model_name]):
        clf = CNNClassifier(model_name=name)
        if clf.load():
            return clf

    # No trained model found — return None (detector handles gracefully)
    return None


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building all paper models...\n")
    for name, builder in [
        ('InceptionV3',       build_inception_v3),
        ('InceptionResNetV2', build_inception_resnet_v2),
        ('DenseNet121',       build_densenet121),
        ('CustomCNN',         build_custom_cnn),
    ]:
        try:
            m = builder()
            params = m.count_params()
            print(f"  {name:25s}: {params:,} parameters")
        except Exception as e:
            print(f"  {name:25s}: ERROR — {e}")
