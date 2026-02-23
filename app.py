import base64
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

try:
    import requests
except Exception:
    requests = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "dataset"
DEFAULT_CAT_JSON = BASE_DIR / "cat_to_name.json"
DEFAULT_MODEL_PATHS = [
    BASE_DIR / "models" / "efficientnetb0_flowers_final.pth",
    BASE_DIR / "models" / "efficientnetb0_flowers_finetuned.pth",
    BASE_DIR / "models" / "efficientnetb0_flowers.pth",
]


def resolve_model_path() -> Path:
    for model_path in DEFAULT_MODEL_PATHS:
        if model_path.exists():
            return model_path
    return DEFAULT_MODEL_PATHS[0]


def model_input_size() -> tuple[int, int]:
    return 224, 224


def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_model = models.efficientnet_b0(weights=weights)
    model = EfficientNetFlowers(base_model, num_classes)
    return model


@lru_cache(maxsize=1)
def load_model():
    model_path = resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Place your trained .pth model at one of: "
            f"{DEFAULT_MODEL_PATHS}"
        )

    class_names = load_class_names()
    model = build_model(len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, model_path, device


@lru_cache(maxsize=1)
def load_label_mapping() -> dict[int, str]:
    if not DEFAULT_CAT_JSON.exists():
        raise FileNotFoundError(f"Missing file: {DEFAULT_CAT_JSON}")

    train_dir = DEFAULT_DATASET_ROOT / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing folder: {train_dir}")

    with open(DEFAULT_CAT_JSON, "r") as file:
        cat_to_name = json.load(file)

    class_dirs = sorted([path.name for path in train_dir.iterdir() if path.is_dir()])
    idx_to_cat_name = {}
    for index, class_dir in enumerate(class_dirs):
        class_id = int(class_dir)
        idx_to_cat_name[index] = cat_to_name.get(str(class_id), f"class_{class_id}")

    return idx_to_cat_name


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    train_dir = DEFAULT_DATASET_ROOT / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing folder: {train_dir}")

    return sorted([path.name for path in train_dir.iterdir() if path.is_dir()])


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


class EfficientNetFlowers(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def description_for_flower(flower_name: str) -> str:
    def detailed_fallback(name: str) -> str:
        return (
            f"{name.title()} is one of the flower categories in the Oxford Flowers-102 benchmark dataset. "
            "In visual classification tasks, this flower can appear quite different across images because of changes "
            "in petal opening stage, camera angle, lighting conditions, background clutter, and natural color variation. "
            "Many flowers in this dataset have overlapping visual patterns, so models must learn fine-grained cues such "
            "as petal shape, petal layering, center structure, and color distribution rather than relying only on one feature. "
            "This prediction means the model found the strongest similarity between the uploaded image and learned examples "
            f"of {name.title()} during training. For practical use, you can also review the Top Predictions list to compare "
            "closely related classes and understand how confident the model is among visually similar flowers."
        )

    if requests is not None:
        key = flower_name.replace(" ", "_")
        candidates = [
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{key}",
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{key}_flower",
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{key}_plant",
        ]
        for url in candidates:
            try:
                response = requests.get(url, timeout=3)
                if response.ok:
                    data = response.json()
                    summary = data.get("extract")
                    if summary and len(summary.strip()) > 120:
                        return summary
            except Exception:
                continue

        wiki_api_urls = [
            "https://en.wikipedia.org/w/api.php",
        ]
        title_candidates = [flower_name, f"{flower_name} flower", f"{flower_name} plant"]
        for api_url in wiki_api_urls:
            for title in title_candidates:
                try:
                    params = {
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": True,
                        "exintro": False,
                        "exchars": 1200,
                        "format": "json",
                        "titles": title,
                    }
                    response = requests.get(api_url, params=params, timeout=4)
                    if not response.ok:
                        continue
                    payload = response.json()
                    pages = payload.get("query", {}).get("pages", {})
                    for page in pages.values():
                        extract = page.get("extract", "").strip()
                        if len(extract) > 220:
                            return extract
                except Exception:
                    continue

    return detailed_fallback(flower_name)


def image_from_request(req) -> Image.Image | None:
    upload = req.files.get("image")
    if upload and upload.filename:
        return Image.open(upload.stream)

    captured = req.form.get("captured_image", "").strip()
    if captured.startswith("data:image") and "," in captured:
        encoded = captured.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes))

    return None


app = Flask(__name__)


@app.get("/")
def home():
    model_path = resolve_model_path()
    return render_template(
        "index.html",
        prediction=None,
        description=None,
        top_predictions=[],
        error=None,
        model_path=str(model_path),
    )


@app.post("/predict")
def predict():
    error = None
    prediction = None
    description = None
    top_predictions = []

    try:
        image = image_from_request(request)
        if image is None:
            raise ValueError("Please upload an image or capture one from camera.")

        model, model_path, device = load_model()
        idx_to_cat_name = load_label_mapping()
        input_size = model_input_size()

        input_tensor = preprocess_image(image, input_size).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]

        best_idx = int(top_indices[0])
        best_name = idx_to_cat_name.get(best_idx, f"class_{best_idx}")
        best_conf = float(probs[best_idx]) * 100

        prediction = f"{best_name.title()} ({best_conf:.2f}%)"
        description = description_for_flower(best_name)

        for rank, idx in enumerate(top_indices, start=1):
            flower = idx_to_cat_name.get(int(idx), f"class_{int(idx)}")
            confidence = float(probs[int(idx)]) * 100
            top_predictions.append(
                {
                    "rank": rank,
                    "flower": flower.title(),
                    "confidence": f"{confidence:.2f}%",
                }
            )

        return render_template(
            "index.html",
            prediction=prediction,
            description=description,
            top_predictions=top_predictions,
            error=None,
            model_path=str(model_path),
        )

    except Exception as exc:
        error = str(exc)
        return render_template(
            "index.html",
            prediction=prediction,
            description=description,
            top_predictions=top_predictions,
            error=error,
            model_path=str(resolve_model_path()),
        )


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
