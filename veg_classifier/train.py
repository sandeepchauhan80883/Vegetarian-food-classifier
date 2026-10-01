"""
train.py
--------
Two-phase transfer-learning training loop:
  Phase 1 (head-only):      train the new classifier head, backbone frozen.
  Phase 2 (fine-tuning):    unfreeze layer4 + head, train end-to-end at a
                             10x lower learning rate.
Best model (by validation accuracy) is checkpointed to
config.CHECKPOINT_PATH. Per-epoch loss/accuracy history is returned for
plotting.
"""

import copy
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import config
from .model import freeze_backbone, unfreeze_for_finetuning, count_trainable_params


def _run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, running_correct, n_samples = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            n_samples += images.size(0)

    return running_loss / n_samples, running_correct / n_samples


def _train_phase(model, train_loader, val_loader, epochs, lr, history,
                  best_state, phase_name, patience=config.EARLY_STOPPING_PATIENCE):
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable, lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_acc)

        history["phase"].append(phase_name)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_state["val_acc"]
        if improved:
            best_state["val_acc"] = val_acc
            best_state["model_state"] = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"[{phase_name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"{'*best*' if improved else ''} ({time.time()-t0:.1f}s)")

        if epochs_no_improve >= patience:
            print(f"[{phase_name}] early stopping (no val improvement for "
                  f"{patience} epochs)")
            break


def train_model(model, train_loader, val_loader,
                 phase1_epochs=config.PHASE1_EPOCHS, phase1_lr=config.PHASE1_LR,
                 phase2_epochs=config.PHASE2_EPOCHS, phase2_lr=config.PHASE2_LR):
    model = model.to(config.DEVICE)
    history = {"phase": [], "epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_state = {"val_acc": 0.0, "model_state": None}

    print(f"Device: {config.DEVICE}")

    print(f"\n=== Phase 1: head-only training "
          f"({count_trainable_params(model):,} trainable params) ===")
    freeze_backbone(model)
    _train_phase(model, train_loader, val_loader, phase1_epochs, phase1_lr,
                 history, best_state, phase_name="phase1_head_only")

    print(f"\n=== Phase 2: fine-tuning layer4 + head "
          f"({count_trainable_params(unfreeze_for_finetuning(model)):,} "
          f"trainable params) ===")
    _train_phase(model, train_loader, val_loader, phase2_epochs, phase2_lr,
                 history, best_state, phase_name="phase2_finetune")

    if best_state["model_state"] is not None:
        model.load_state_dict(best_state["model_state"])
    torch.save(model.state_dict(), config.CHECKPOINT_PATH)
    print(f"\nBest val_acc={best_state['val_acc']:.4f} -- "
          f"checkpoint saved to {config.CHECKPOINT_PATH}")

    return model, history
