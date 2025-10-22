"""
Utility functions for training, evaluation, and visualization
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, confusion_matrix
import seaborn as sns


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute classification metrics.
    
    Args:
        y_true: true labels (N,)
        y_pred: predicted labels (N,)
        y_prob: predicted probabilities for positive class (N,) - optional for AUC
    
    Returns:
        dict with 'accuracy', 'f1', and optionally 'auc'
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 score
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Handle case where only one class present
            metrics['auc'] = 0.0
    
    return metrics


def save_learning_curves(train_losses, val_metrics, save_path):
    """
    Plot and save learning curves.
    
    Args:
        train_losses: list of training losses per epoch
        val_metrics: list of dicts with validation metrics per epoch
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    axes[0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Contrastive Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot validation metrics
    if val_metrics:
        val_aucs = [m.get('auc', 0) for m in val_metrics]
        val_accs = [m.get('accuracy', 0) for m in val_metrics]
        val_f1s = [m.get('f1', 0) for m in val_metrics]
        
        axes[1].plot(epochs, val_aucs, 'r-', linewidth=2, label='AUC', marker='o')
        axes[1].plot(epochs, val_accs, 'g-', linewidth=2, label='Accuracy', marker='s')
        axes[1].plot(epochs, val_f1s, 'orange', linewidth=2, label='F1', marker='^')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to {save_path}")


def save_roc_curve(y_true, y_prob, save_path):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: true labels (N,)
        y_prob: predicted probabilities (N,)
        save_path: path to save figure
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"ROC curve saved to {save_path}")
    except Exception as e:
        print(f"Could not save ROC curve: {e}")


def save_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: true labels (N,)
        y_pred: predicted labels (N,)
        save_path: path to save figure
        class_names: optional list of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_embeddings_umap(embeddings, labels, save_path):
    """
    Plot UMAP projection of embeddings (optional, requires umap-learn).
    
    Args:
        embeddings: embedding vectors (N, D)
        labels: class labels (N,)
        save_path: path to save figure
    """
    try:
        import umap
        
        # Fit UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                            c=labels, cmap='coolwarm', s=10, alpha=0.6)
        plt.colorbar(scatter, label='Class Label')
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.title('UMAP Projection of Embeddings', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"UMAP embedding plot saved to {save_path}")
    except ImportError:
        print("UMAP not installed. Skipping embedding visualization.")
    except Exception as e:
        print(f"Could not create UMAP plot: {e}")


def save_metrics_json(metrics, save_path):
    """Save metrics dictionary to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")


def create_output_dirs(out_dir):
    """Create output directory structure."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    print(f"Output directories created at {out_dir}")