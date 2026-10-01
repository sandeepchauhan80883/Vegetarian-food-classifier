"""
dataset.py
----------
Preprocessing pipeline + DataLoader construction.

Train-time transforms include data augmentation (random crop, flip, colour
jitter, rotation) to reduce overfitting on a modest-sized food-image
dataset. Val/test transforms are deterministic (resize + center-crop) so
evaluation numbers are stable and comparable across runs. All splits are
normalised with ImageNet mean/std since the backbone was pretrained on
ImageNet.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from . import config


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(config.IMAGE_SIZE * 1.15)),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_dataloaders(batch_size=None, num_workers=None):
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    train_tf, eval_tf = build_transforms()

    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(config.VAL_DIR, transform=eval_tf)
    test_ds = datasets.ImageFolder(config.TEST_DIR, transform=eval_tf)

    # Sanity check: ImageFolder's alphabetical class order should match
    # config.CLASSES exactly, since everything downstream (meal-category
    # lookup, reporting) indexes by that order.
    assert train_ds.classes == config.CLASSES, (
        f"Class order mismatch.\nImageFolder: {train_ds.classes}\n"
        f"config.CLASSES: {config.CLASSES}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)
