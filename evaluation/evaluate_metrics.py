"""
Evaluation Metrics Utilities — PyTorch & Scikit-learn Compatible
=================================================================
Framework-agnostic functions for model evaluation. All functions accept numpy
arrays (predictions & ground truth) rather than model objects, making them
usable with PyTorch, TensorFlow, or scikit-learn models.

Functions:
    compute_metrics          — accuracy, precision, recall, F1, top-5 accuracy
    plot_confusion_matrix    — seaborn heatmap confusion matrix
    print_classification_report — sklearn classification report
    plot_roc_curve_multiclass   — one-vs-rest ROC curves
    plot_pr_curve_multiclass    — precision-recall curves
    plot_training_history       — loss & accuracy learning curves
    plot_model_comparison       — grouped bar chart comparing models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
import time
from typing import Optional


# ── Core Metrics Computation ─────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> dict:
    """
    Compute classification metrics from numpy arrays.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (integer-encoded), shape (N,).
    y_pred : np.ndarray
        Predicted labels (integer-encoded), shape (N,).
    y_proba : np.ndarray, optional
        Predicted probability matrix, shape (N, C). Required for top-5 accuracy.
    model_name : str
        Name for display purposes.

    Returns
    -------
    dict
        Keys: model, accuracy, precision, recall, f1, top5_accuracy (if y_proba given).
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    result = {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

    if y_proba is not None and y_proba.ndim == 2:
        top5_acc = _top_k_accuracy(y_true, y_proba, k=5)
        result['top5_accuracy'] = top5_acc

    return result


def _top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from probability matrix."""
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: tuple = (20, 18),
    normalize: bool = False,
) -> np.ndarray:
    """
    Plots a heatmap confusion matrix.

    Parameters
    ----------
    y_true : array-like — True labels (integer-encoded).
    y_pred : array-like — Predicted labels (integer-encoded).
    classes : list of str — Class names in label order.
    title : str — Plot title.
    cmap : str — Matplotlib colormap name.
    figsize : tuple — Figure size.
    normalize : bool — If True, normalize by true class counts.

    Returns
    -------
    np.ndarray — The confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=False, cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()
    return cm


# ── Classification Report ───────────────────────────────────────────────────

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
) -> str:
    """
    Prints and returns the sklearn classification report string.
    """
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    present_names = [classes[int(i)] for i in present_labels if int(i) < len(classes)]
    report = classification_report(
        y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0
    )
    print("Classification Report:\n")
    print(report)
    return report


# ── ROC Curve (Multi-class) ──────────────────────────────────────────────────

def plot_roc_curve_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: list,
    title: str = 'ROC Curve (Multi-class)',
    figsize: tuple = (14, 10),
):
    """
    Plots one-vs-rest ROC curves for each class.
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    valid_aucs = []
    for i in range(n_classes):
        if y_true_bin.shape[1] > i:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            score = auc(fpr[i], tpr[i])
            roc_auc[i] = 0.0 if np.isnan(score) else score
            if not np.isnan(score):
                valid_aucs.append(score)
        else:
            fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), 0.0

    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    cmap = plt.colormaps['tab20']
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(n_classes):
        color = cmap(i / max(1, n_classes))
        ax.plot(fpr[i], tpr[i], color=color, lw=1,
                label=f'{classes[i]} (AUC={roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{title}  (Macro AUC = {mean_auc:.4f})', fontsize=14)
    ax.legend(loc='lower right', fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()


# ── PR Curve (Multi-class) ──────────────────────────────────────────────────

def plot_pr_curve_multiclass(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: list,
    title: str = 'Precision-Recall Curve (Multi-class)',
    figsize: tuple = (14, 10),
):
    """
    Plots one-vs-rest Precision-Recall curves for each class.
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    cmap = plt.colormaps['tab20']
    fig, ax = plt.subplots(figsize=figsize)

    mean_ap = []
    for i in range(n_classes):
        if y_true_bin.shape[1] > i:
            precision_i, recall_i, _ = precision_recall_curve(
                y_true_bin[:, i], y_score[:, i]
            )
            ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
            ap = 0.0 if np.isnan(ap) else ap
        else:
            precision_i, recall_i, ap = np.array([0, 1]), np.array([0, 0]), 0.0

        mean_ap.append(ap)
        color = cmap(i / max(1, n_classes))
        ax.plot(recall_i, precision_i, color=color, lw=1,
                label=f'{classes[i]} (AP={ap:.2f})')

    macro_ap = np.mean(mean_ap)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{title}  (Macro AP = {macro_ap:.4f})', fontsize=14)
    ax.legend(loc='lower left', fontsize='x-small', ncol=2)
    plt.tight_layout()
    plt.show()


# ── Training History (Learning Curves) ───────────────────────────────────────

def plot_training_history(history: dict, model_name: str = 'Model'):
    """
    Plots accuracy and loss curves from a training history dictionary.

    Parameters
    ----------
    history : dict
        Expected keys: 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
        Also supports Keras-style keys: 'accuracy', 'val_accuracy', 'loss', 'val_loss'.
    model_name : str — Name for the plot title.
    """
    # Support both PyTorch-style and Keras-style keys
    if hasattr(history, 'history'):
        history = history.history

    train_acc = history.get('train_acc', history.get('accuracy', []))
    val_acc = history.get('val_acc', history.get('val_accuracy', []))
    train_loss = history.get('train_loss', history.get('loss', []))
    val_loss = history.get('val_loss', history.get('val_loss', []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    if train_acc:
        ax1.plot(train_acc, label='Train Accuracy', marker='o', markersize=3)
    if val_acc:
        ax1.plot(val_acc, label='Validation Accuracy', marker='s', markersize=3)
    ax1.set_title(f'{model_name} — Accuracy', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    if train_loss:
        ax2.plot(train_loss, label='Train Loss', marker='o', markersize=3)
    if val_loss:
        ax2.plot(val_loss, label='Validation Loss', marker='s', markersize=3)
    ax2.set_title(f'{model_name} — Loss', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── Model Comparison Chart ───────────────────────────────────────────────────

def plot_model_comparison(results_list: list, figsize: tuple = (14, 6)):
    """
    Plots a grouped bar chart comparing multiple models.

    Parameters
    ----------
    results_list : list of dict
        Each dict should have keys: model, accuracy, precision, recall, f1.
    figsize : tuple — Figure size.
    """
    import pandas as pd

    df = pd.DataFrame(results_list)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    x = np.arange(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=figsize)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if metric in df.columns:
            ax.bar(x + i * width, df[metric], width, label=label)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['model'], rotation=30, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
