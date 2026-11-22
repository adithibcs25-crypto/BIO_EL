"""Visualization module for evaluation figures and immune optimization dynamics."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


# Set high-contrast dark theme styling for crisp readability on dark backgrounds
plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#0b0f19"
plt.rcParams["axes.facecolor"] = "#111827"
plt.rcParams["axes.edgecolor"] = "#374151"
plt.rcParams["axes.labelcolor"] = "#ffffff"
plt.rcParams["text.color"] = "#ffffff"
plt.rcParams["xtick.color"] = "#ffffff"
plt.rcParams["ytick.color"] = "#ffffff"
plt.rcParams["grid.color"] = "#1f2937"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def plot_confusion_matrices(
    metrics_list: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot side-by-side confusion matrix heatmaps for all evaluated models."""
    n_models = len(metrics_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)

    for idx, m in enumerate(metrics_list):
        ax = axes[0, idx]
        cm = np.array(m["confusion_matrix"])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Annotate with both count and percentage
        annot = np.empty_like(cm).astype(str)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues" if idx == 0 else "Greens",
            cbar=False,
            ax=ax,
            xticklabels=["Normal (0)", "Attack (1)"],
            yticklabels=["Normal (0)", "Attack (1)"],
            linewidths=1,
            linecolor="white",
        )
        ax.set_title(f"{m['model_name']}\nAccuracy: {m['accuracy']:.3%}", fontsize=13, weight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_metrics_comparison(
    metrics_list: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot grouped bar chart comparing performance metrics across models."""
    records = []
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall (DR)", "recall"),
        ("Specificity", "specificity"),
        ("F1-Score", "f1_score"),
        ("FPR", "false_positive_rate"),
        ("MCC", "mcc"),
    ]

    for m in metrics_list:
        name = m["model_name"]
        for label, key in metric_keys:
            records.append({
                "Model": name,
                "Metric": label,
                "Value": m[key]
            })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    sns.barplot(
        data=df,
        x="Metric",
        y="Value",
        hue="Model",
        palette=palette[:len(metrics_list)],
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score / Rate", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("NIDS Performance Comparison: AIS vs. Baseline", fontsize=14, weight="bold", pad=12)
    ax.legend(title="Model", frameon=True, loc="upper right")

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0.001:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                color="white",
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_clonal_progress(
    history: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[plt.Figure]:
    """Plot Clonal Selection optimization progress over generations."""
    if not history:
        return None

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Affinity Maturation
    ax1 = axes[0]
    ax1.plot(df["generation"], df["best_affinity"], marker="o", color="#2ca02c", lw=2, label="Best Affinity")
    ax1.plot(df["generation"], df["mean_affinity"], marker="s", color="#1f77b4", lw=2, linestyle="--", label="Mean Affinity")
    ax1.set_title("Immune Affinity Maturation Across Generations", fontsize=12, weight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Affinity Score")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: TP vs FP Trends
    ax2 = axes[1]
    ax2.plot(df["generation"], df["mean_tp"], marker="^", color="#1f77b4", lw=2, label="Mean True Positives (TP)")
    ax2.plot(df["generation"], df["mean_fp"], marker="v", color="#d62728", lw=2, label="Mean False Positives (FP)")
    ax2.set_title("Detector Discrimination Balance (TP vs FP)", fontsize=12, weight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Average Count per Detector")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot Receiver Operating Characteristic (ROC) curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, (name, model) in enumerate(models_dict.items()):
        color = colors[idx % len(colors)]
        scores = None
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "score_samples"):
            scores = model.score_samples(X_test)

        if scores is not None:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    ax.set_title("ROC Curves for Intrusion Detection Models", fontsize=13, weight="bold")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_execution_times(
    metrics_list: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training and inference latency comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    names = [m["model_name"] for m in metrics_list]
    train_times = [m["training_time_sec"] for m in metrics_list]
    pred_times = [m["prediction_time_sec"] for m in metrics_list]

    # Training Time
    axes[0].bar(names, train_times, color=["#3274a1", "#3b9ab2"], edgecolor="black", width=0.5)
    axes[0].set_title("Training Duration (Seconds)", fontsize=12, weight="bold")
    axes[0].set_ylabel("Seconds")
    for i, v in enumerate(train_times):
        axes[0].text(i, v + (max(train_times) * 0.02), f"{v:.2f}s", ha="center", weight="bold", color="white")

    # Prediction Latency
    axes[1].bar(names, pred_times, color=["#e1812c", "#e76f51"], edgecolor="black", width=0.5)
    axes[1].set_title("Inference Latency on Test Set (Seconds)", fontsize=12, weight="bold")
    axes[1].set_ylabel("Seconds")
    for i, v in enumerate(pred_times):
        axes[1].text(i, v + (max(pred_times) * 0.02), f"{v:.2f}s", ha="center", weight="bold", color="white")

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
