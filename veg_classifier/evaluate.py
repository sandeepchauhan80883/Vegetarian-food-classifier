"""
evaluate.py
-----------
Turns a trained model + test set into the deliverables a project report
needs: overall metrics, a confusion matrix, per-meal-category performance
(the project's specific ask), sample correct predictions, and a gallery of
misclassified examples for error analysis.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

from . import config


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(config.DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_probs))


def overall_report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report_txt = classification_report(
        y_true, y_pred, target_names=config.CLASSES, digits=3, zero_division=0
    )
    print(f"Overall test accuracy: {acc:.4f}   Macro F1: {macro_f1:.4f}\n")
    print(report_txt)
    return acc, macro_f1, report_txt


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASSES, rotation=90, fontsize=8)
    ax.set_yticklabels(config.CLASSES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix -- Vegetarian Dish Classification")
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return cm


def meal_category_performance(y_true, y_pred, save_path):
    """The project's specific ask: accuracy broken down by meal category
    (Breakfast / Snack / Main Course / Dessert), not just overall accuracy."""
    cats_true = np.array([config.IDX_TO_MEAL_CATEGORY[i] for i in y_true])
    cats_pred = np.array([config.IDX_TO_MEAL_CATEGORY[i] for i in y_pred])

    rows = []
    for cat in config.MEAL_CATEGORIES:
        mask = cats_true == cat
        n = mask.sum()
        # accuracy at the fine-grained dish level, restricted to this category
        dish_acc = accuracy_score(y_true[mask], y_pred[mask]) if n else float("nan")
        # accuracy at the coarser "did we at least get the meal category right" level
        cat_acc = (cats_pred[mask] == cat).mean() if n else float("nan")
        rows.append((cat, n, dish_acc, cat_acc))

    print(f"{'Meal Category':<15}{'#Test imgs':<12}{'Dish Acc':<12}{'Category Acc':<12}")
    for cat, n, dish_acc, cat_acc in rows:
        print(f"{cat:<15}{n:<12}{dish_acc:<12.3f}{cat_acc:<12.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = [r[0] for r in rows]
    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width / 2, [r[2] for r in rows], width, label="Exact-dish accuracy")
    ax.bar(x + width / 2, [r[3] for r in rows], width, label="Meal-category accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance by Meal Category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return rows


def _denormalize(img_tensor):
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_sample_grid(images, y_true, y_pred, probs, save_path, title, n=12):
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        ax.imshow(_denormalize(images[i]))
        true_c = config.CLASSES[y_true[i]]
        pred_c = config.CLASSES[y_pred[i]]
        conf = probs[i][y_pred[i]]
        correct = y_true[i] == y_pred[i]
        color = "green" if correct else "red"
        ax.set_title(f"true: {true_c}\npred: {pred_c} ({conf:.2f})",
                     fontsize=8, color=color)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def sample_outputs_and_errors(model, test_loader, test_dataset, artifacts_dir):
    """Generates: (a) a grid of sample predictions, (b) a gallery of
    misclassified examples for error analysis."""
    model.eval()
    images_batch, labels_batch = next(iter(test_loader))
    images_batch = images_batch.to(config.DEVICE)
    outputs = model(images_batch)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    preds = probs.argmax(1)

    plot_sample_grid(
        images_batch, labels_batch.numpy(), preds, probs,
        os.path.join(artifacts_dir, "sample_predictions.png"),
        title="Sample Predictions (green = correct, red = incorrect)",
    )

    # Now scan the FULL test set for misclassified examples (richer error analysis)
    mis_images, mis_true, mis_pred, mis_probs = [], [], [], []
    for images, labels in test_loader:
        images_d = images.to(config.DEVICE)
        outputs = model(images_d)
        p = torch.softmax(outputs, dim=1).cpu().numpy()
        pr = p.argmax(1)
        wrong = pr != labels.numpy()
        if wrong.any():
            mis_images.append(images[wrong])
            mis_true.append(labels.numpy()[wrong])
            mis_pred.append(pr[wrong])
            mis_probs.append(p[wrong])
        if sum(x.shape[0] for x in mis_images) >= 12:
            break

    if mis_images:
        mis_images = torch.cat(mis_images)[:12]
        mis_true = np.concatenate(mis_true)[:12]
        mis_pred = np.concatenate(mis_pred)[:12]
        mis_probs = np.concatenate(mis_probs)[:12]
        plot_sample_grid(
            mis_images, mis_true, mis_pred, mis_probs,
            os.path.join(artifacts_dir, "misclassified_examples.png"),
            title="Misclassified Examples -- Error Analysis", n=len(mis_images),
        )
        print(f"Saved {len(mis_images)} misclassified examples for error analysis.")
    else:
        print("No misclassified examples found in the scanned batches "
              "(model fit this test set perfectly, or dataset is very small).")
