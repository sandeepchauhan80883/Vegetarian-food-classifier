"""
model.py
--------
Transfer-learning model: ImageNet-pretrained ResNet50 backbone with its
final fully-connected layer replaced by a small classifier head sized to
our vegetarian-dish classes.
"""

import warnings
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from . import config


def build_model(num_classes=config.NUM_CLASSES, dropout=0.3):
    """Builds ResNet50 with ImageNet weights when reachable, falling back to
    random initialisation with a clear warning if pretrained weights can't
    be downloaded (e.g. no internet in a sandboxed environment). In a normal
    training environment (Colab / local machine with internet) this will
    silently succeed and give you the full benefit of transfer learning."""
    try:
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        pretrained_loaded = True
    except Exception as e:
        warnings.warn(
            f"Could not download ImageNet-pretrained weights ({e}). "
            "Falling back to random initialisation -- this run is only "
            "useful as a pipeline smoke-test, not a real result. Re-run "
            "with normal internet access to get true transfer learning."
        )
        backbone = models.resnet50(weights=None)
        pretrained_loaded = False

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    freeze_backbone(backbone)  # start in phase-1 (head-only) mode
    return backbone, pretrained_loaded


def freeze_backbone(model):
    """Phase 1: freeze every layer except the new classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_for_finetuning(model, unfreeze_from="layer4"):
    """Phase 2: unfreeze the last residual block (+ head) for end-to-end
    fine-tuning at a lower learning rate. Earlier low-level-feature layers
    (layer1/layer2/layer3) stay frozen since low-level edge/texture filters
    transfer well from ImageNet and re-training them risks overfitting on a
    modest-sized food dataset."""
    unfreeze = False
    for name, param in model.named_parameters():
        if name.startswith(unfreeze_from) or name.startswith("fc."):
            unfreeze = True
        param.requires_grad = unfreeze or name.startswith("fc.")
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
