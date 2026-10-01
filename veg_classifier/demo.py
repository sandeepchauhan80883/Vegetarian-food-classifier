"""
demo.py
-------
The "working demonstration" deliverable: load the trained checkpoint and
classify a single new image, returning the top-k vegetarian dish
predictions and their meal category. Also exposes an optional Gradio
web app (`launch_gradio_demo()`) for an interactive upload-and-classify UI.
"""

import torch
from PIL import Image

from . import config
from .model import build_model
from .dataset import build_transforms


def load_trained_model(checkpoint_path=config.CHECKPOINT_PATH):
    model, _ = build_model()
    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path_or_pil, top_k=3):
    """Returns a list of (dish_name, meal_category, probability) tuples,
    sorted by descending probability."""
    _, eval_tf = build_transforms()

    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    x = eval_tf(img).unsqueeze(0).to(config.DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_probs, top_idx = probs.topk(top_k)
    results = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        dish = config.CLASSES[idx]
        meal_cat = config.IDX_TO_MEAL_CATEGORY[idx]
        results.append((dish, meal_cat, p))
    return results


def launch_gradio_demo(checkpoint_path=config.CHECKPOINT_PATH):
    """Optional interactive web demo. Run `python -m veg_classifier.demo`
    to launch it locally (needs `pip install gradio`)."""
    import gradio as gr

    model = load_trained_model(checkpoint_path)

    def classify(image):
        results = predict_image(model, image, top_k=5)
        return {f"{dish}  ({cat})": prob for dish, cat, prob in results}

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type="pil", label="Upload a food photo"),
        outputs=gr.Label(num_top_classes=5, label="Predicted vegetarian dish"),
        title="Vegetarian Food Image Classifier",
        description=(
            "ResNet50 transfer-learning model. Classifies a mobile food photo "
            "into one of "
            f"{config.NUM_CLASSES} vegetarian dishes across "
            f"{len(config.MEAL_CATEGORIES)} meal categories: "
            f"{', '.join(config.MEAL_CATEGORIES)}."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    launch_gradio_demo()
