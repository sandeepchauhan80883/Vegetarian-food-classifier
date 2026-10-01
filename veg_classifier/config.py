"""
config.py
---------
Single source of truth for class list, meal-category grouping, paths and
hyperparameters. Editing THIS file is enough to point the whole pipeline at
a different dataset (e.g. swap the Food-101 vegetarian subset below for the
Kaggle "Indian Food Images" vegetarian subset - see README.md, Section 1).
"""

import os
import torch

# ---------------------------------------------------------------------------
# 1. CLASSES  +  MEAL-CATEGORY MAPPING
# ---------------------------------------------------------------------------
# 20 vegetarian dishes curated from the Food-101 dataset (101 classes total).
# Classes containing meat / poultry / fish / seafood / egg-as-main-ingredient
# were excluded. 5 classes were kept per meal category so that per-category
# evaluation (a key deliverable) is on balanced ground.
#
# NOTE (dataset limitation, worth a line in your report): Food-101 labels are
# by dish name, not by verified ingredients, so a handful of real photos
# (e.g. a "pizza" with pepperoni) can be mislabeled with respect to strict
# vegetarian status. For a stricter guarantee, use the Kaggle Indian Food
# Images dataset instead (see README.md) which ships an explicit veg/non-veg
# column per dish.

MEAL_CATEGORY_MAP = {
    # Breakfast
    "pancakes":            "Breakfast",
    "waffles":              "Breakfast",
    "french_toast":         "Breakfast",
    "beignets":              "Breakfast",
    "donuts":                "Breakfast",
    # Snack / Starter
    "garlic_bread":          "Snack",
    "hummus":                 "Snack",
    "bruschetta":             "Snack",
    "onion_rings":            "Snack",
    "nachos":                  "Snack",
    # Main Course
    "pizza":                   "Main Course",
    "macaroni_and_cheese":      "Main Course",
    "risotto":                   "Main Course",
    "gnocchi":                    "Main Course",
    "falafel":                     "Main Course",
    # Dessert
    "cheesecake":                  "Dessert",
    "tiramisu":                     "Dessert",
    "chocolate_cake":                "Dessert",
    "ice_cream":                       "Dessert",
    "baklava":                          "Dessert",
}

CLASSES = sorted(MEAL_CATEGORY_MAP.keys())
NUM_CLASSES = len(CLASSES)
MEAL_CATEGORIES = sorted(set(MEAL_CATEGORY_MAP.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_MEAL_CATEGORY = {CLASS_TO_IDX[c]: cat for c, cat in MEAL_CATEGORY_MAP.items()}

# ---------------------------------------------------------------------------
# 2. PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pt")

# ---------------------------------------------------------------------------
# 3. HYPERPARAMETERS
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 1

# Two-phase transfer-learning schedule (standard recipe):
#   Phase 1: freeze the ResNet backbone, train only the new classifier head.
#   Phase 2: unfreeze the last residual block (layer4) and fine-tune end to
#            end at a much lower learning rate.
PHASE1_EPOCHS = 6
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 6
PHASE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
