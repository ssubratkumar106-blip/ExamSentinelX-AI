"""
ai/evaluate.py
==============
Evaluation & Results Generation
Paper: Section IV — Results and Discussion

Generates ALL paper figures and tables:
  - Confusion Matrix (Fig 5, 7, 9, 11, 15)
  - ROC Curve per class (Fig 5, 7, 9, 11)
  - Accuracy/Loss curves (Fig 4, 6, 8, 10, 16)
  - Classification Report — Precision, Recall, F1 (Tables 4–8)
  - Model comparison table (Table 9)
  - mAP for YOLO (Fig 12–14, Table 8)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from pathlib import Path


CLASS_NAMES = [
    'external_device',
    'head_movement',
    'multiple_persons',
    'talking_to_others',
    'normal'
]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str):
    """
    Plot and save confusion matrix.
    Paper: Fig 5 (DenseNet), Fig 7 (IRNv2), Fig 9 (InceptionV3), Fig 11 (CNN), Fig 15 (YOLO)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Count')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].tick_params(axis='x', rotation=45)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title('Normalized')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    out = Path(save_dir) / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Saved confusion matrix → {out}")
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
# ROC CURVE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve(y_true, y_prob, model_name: str, save_dir: str):
    """
    Plot multi-class ROC curves (one per class).
    Paper: Fig 5, 7, 9, 11
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))

    colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6']
    fig, ax = plt.subplots(figsize=(8, 6))

    aucs = []
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {model_name}\nMean AUC = {np.mean(aucs):.3f}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Saved ROC curve → {out}")
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
# ACCURACY / LOSS CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history_dict: dict, model_name: str, save_dir: str):
    """
    Plot training accuracy and loss curves.
    Paper: Fig 4 (DenseNet), Fig 6 (IRNv2), Fig 8 (InceptionV3), Fig 10 (CNN)

    history_dict example:
      {'accuracy': [...], 'val_accuracy': [...], 'loss': [...], 'val_loss': [...]}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History — {model_name}', fontsize=14, fontweight='bold')

    epochs = range(1, len(history_dict['accuracy']) + 1)

    # Accuracy
    axes[0].plot(epochs, history_dict['accuracy'], '#3b82f6', lw=2, label='Train Accuracy')
    axes[0].plot(epochs, history_dict['val_accuracy'], '#10b981', lw=2,
                 linestyle='--', label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Loss
    axes[1].plot(epochs, history_dict['loss'], '#ef4444', lw=2, label='Train Loss')
    axes[1].plot(epochs, history_dict['val_loss'], '#f59e0b', lw=2,
                 linestyle='--', label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / f'training_history_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Saved training history → {out}")
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_classification_report(y_true, y_pred, model_name: str, save_dir: str) -> dict:
    """
    Generate precision, recall, F1 report.
    Paper: Tables 4, 5, 6, 7
    """
    from sklearn.metrics import classification_report, accuracy_score

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    accuracy = accuracy_score(y_true, y_pred)

    # Print table
    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name}")
    print(f"{'='*60}")
    header = f"{'Class':<25} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Support':>8}"
    print(header)
    print('-' * 65)
    for cls in CLASS_NAMES:
        if cls in report:
            r = report[cls]
            print(f"{cls:<25} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                  f"{r['f1-score']:>10.3f} {r['support']:>8}")
    print('-' * 65)
    print(f"{'Accuracy':<25} {accuracy:>10.3f}")
    print(f"{'Macro Avg':<25} {report['macro avg']['precision']:>10.3f} "
          f"{report['macro avg']['recall']:>8.3f} "
          f"{report['macro avg']['f1-score']:>10.3f}")
    print(f"{'Weighted Avg':<25} {report['weighted avg']['precision']:>10.3f} "
          f"{report['weighted avg']['recall']:>8.3f} "
          f"{report['weighted avg']['f1-score']:>10.3f}")

    # Save as JSON
    report['model_name'] = model_name
    report['accuracy'] = accuracy
    out = Path(save_dir) / f'classification_report_{model_name.lower().replace(" ", "_")}.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[Evaluate] Saved classification report → {out}")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE (Paper Table 9)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comparison_table(results: dict, save_dir: str):
    """
    Generate the model comparison table (Paper Table 9).

    results format:
      {
        'DenseNet121':       {'accuracy': 0.85, 'precision': 0.87, 'recall': 0.85, 'f1': 0.86},
        'InceptionResNetV2': {...},
        ...
      }
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    rows = []
    for model_name, metrics in results.items():
        rows.append([
            model_name,
            f"{metrics.get('accuracy', 0):.3f}",
            f"{metrics.get('precision', 0):.3f}",
            f"{metrics.get('recall', 0):.3f}",
            f"{metrics.get('f1', 0):.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#1e3a5f')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best row
    if rows:
        best_idx = max(range(len(rows)),
                       key=lambda i: float(rows[i][1]))
        for j in range(len(headers)):
            table[(best_idx + 1, j)].set_facecolor('#10b981')
            table[(best_idx + 1, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Table 9: Performance Comparison of All Models',
                 fontweight='bold', pad=20, fontsize=13)

    plt.tight_layout()
    out = Path(save_dir) / 'model_comparison_table9.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Evaluate] Saved comparison table → {out}")
    return str(out)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_keras_model(
    model,
    test_dataset,
    model_name: str,
    history=None,
    save_dir: str = 'results'
) -> dict:
    """
    Full evaluation pipeline for a Keras model.
    Generates all plots and reports for the paper.

    Args:
        model:        Trained Keras model
        test_dataset: tf.data.Dataset or (X_test, y_test) tuple
        model_name:   Display name (e.g. 'DenseNet121')
        history:      Keras training History object (for accuracy/loss plots)
        save_dir:     Directory to save all output files

    Returns:
        Dict with all metrics.
    """
    import tensorflow as tf

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # --- Get predictions ---
    print(f"\n[Evaluate] Evaluating {model_name}...")

    if isinstance(test_dataset, tuple):
        X_test, y_true = test_dataset
        y_prob = model.predict(X_test, verbose=0)
    else:
        # tf.data.Dataset
        y_true_list, y_prob_list = [], []
        for X_batch, y_batch in test_dataset:
            y_prob_list.append(model.predict(X_batch, verbose=0))
            y_true_list.append(np.argmax(y_batch.numpy(), axis=1))
        y_prob = np.concatenate(y_prob_list)
        y_true = np.concatenate(y_true_list)

    y_pred = np.argmax(y_prob, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    # --- Generate all paper figures ---
    plot_confusion_matrix(y_true, y_pred, model_name, save_dir)
    plot_roc_curve(y_true, y_prob, model_name, save_dir)
    report = generate_classification_report(y_true, y_pred, model_name, save_dir)

    if history is not None:
        hist_dict = {
            'accuracy': history.history.get('accuracy', []),
            'val_accuracy': history.history.get('val_accuracy', []),
            'loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', []),
        }
        plot_training_history(hist_dict, model_name, save_dir)

    results = {
        'model': model_name,
        'accuracy': float(report.get('accuracy', 0)),
        'precision': float(report.get('weighted avg', {}).get('precision', 0)),
        'recall':    float(report.get('weighted avg', {}).get('recall', 0)),
        'f1':        float(report.get('weighted avg', {}).get('f1-score', 0)),
    }
    print(f"[Evaluate] {model_name}: Acc={results['accuracy']:.3f} "
          f"P={results['precision']:.3f} R={results['recall']:.3f} F1={results['f1']:.3f}")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper evaluation figures')
    parser.add_argument('--results-dir', default='results/evaluation',
                        help='Directory containing saved model result JSONs')
    parser.add_argument('--output-dir', default='results/figures',
                        help='Directory to save generated figures')
    args = parser.parse_args()

    # Load any existing result JSONs and generate comparison table
    results_dir = Path(args.results_dir)
    all_results = {}

    if results_dir.exists():
        for json_file in results_dir.glob('classification_report_*.json'):
            with open(json_file) as f:
                data = json.load(f)
            model_name = data.get('model_name', json_file.stem)
            all_results[model_name] = {
                'accuracy': data.get('accuracy', 0),
                'precision': data.get('weighted avg', {}).get('precision', 0),
                'recall':    data.get('weighted avg', {}).get('recall', 0),
                'f1':        data.get('weighted avg', {}).get('f1-score', 0),
            }

    if all_results:
        generate_comparison_table(all_results, args.output_dir)
        print(f"\nGenerated comparison table with {len(all_results)} models.")
    else:
        print("No result JSONs found. Train models first with: python ai/train_model.py")
