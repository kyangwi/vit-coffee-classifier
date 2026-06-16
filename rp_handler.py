import base64
import io
import json
import os
import urllib.request
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

try:
    import runpod
except ImportError:
    runpod = None

# Model Configuration
IMG_SIZE = 224
DEFAULT_CLASSES = ["KR1", "KR10", "KR3", "KR4", "KR5", "KR6", "KR7", "KR8", "KR9"]
MODEL_FILENAME = "vit_base_patch16_224_coffee_preloaded.pth"

class PyTorchViTModel(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Dropout(p=0.25),
            nn.Linear(768, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
classes = DEFAULT_CLASSES

def load_class_names() -> list[str]:
    # Check if a custom class_names.json is available in the current directory
    if os.path.exists("class_names.json"):
        try:
            with open("class_names.json", "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and all(isinstance(n, str) for n in names):
                return names
        except Exception as e:
            print(f"Error loading custom class_names.json: {e}")
    return DEFAULT_CLASSES

def remap_fastai_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.removeprefix("module.")

        if key.startswith("0.model."):
            new_key = key.replace("0.model.", "backbone.", 1)
        elif key.startswith("1."):
            new_key = key.replace("1.", "head.", 1)
        else:
            new_key = key

        remapped[new_key] = value
    return remapped

def load_model() -> nn.Module:
    global model, classes
    if model is not None:
        return model

    classes = load_class_names()
    
    # Path of the model file
    model_path = os.environ.get("MODEL_PATH", MODEL_FILENAME)
    
    # If the model is not found locally, download from HF Hub
    if not os.path.exists(model_path):
        repo_id = os.environ.get("HF_REPO_ID", "Bwenge840/vit-base-patch16-224-coffee-preloaded")
        filename = os.environ.get("HF_FILENAME", "vit_base_patch16_224_coffee_preloaded.pth")
        print(f"Model file '{model_path}' not found locally. Attempting HF download: {repo_id}/{filename}...")
        
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            model_path = downloaded
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            # Try loading whatever is available in the workspace if we fail
            if os.path.exists(MODEL_FILENAME):
                model_path = MODEL_FILENAME
            else:
                raise FileNotFoundError(f"ViT model file could not be resolved. Error: {e}")

    print(f"Loading ViT model weights from: {model_path} on device: {device}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("Unsupported checkpoint format.")

        model = PyTorchViTModel(num_classes=len(classes))
        model.load_state_dict(remap_fastai_state_dict(state_dict))

    model.to(device)
    model.eval()
    print("ViT model loaded successfully and set to evaluation mode.")
    return model

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((IMG_SIZE, IMG_SIZE), resampling)

    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transform(image_tensor).unsqueeze(0).to(device)

def handler(job):
    """
    RunPod serverless handler function.
    Expected job input:
    {
        "input": {
            "image_base64": "<base64_string>",  # Choice 1
            "image_url": "<url_string>",        # Choice 2
            "top_k": 3                          # Optional, default 3
        }
    }
    """
    job_input = job.get("input", {})
    image_base64 = job_input.get("image_base64")
    image_url = job_input.get("image_url")
    top_k = job_input.get("top_k", 3)

    if not image_base64 and not image_url:
        return {"error": "Invalid request. Please provide 'image_base64' or 'image_url' in job inputs."}

    try:
        # Resolve model loading on first call
        load_model()

        # Retrieve bytes
        if image_base64:
            # Strip potential data URL prefix
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(image_base64)
        else:
            # Download from url
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(image_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                image_bytes = response.read()

        image_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_k = max(1, min(top_k, len(classes)))
        top_probs, top_idxs = torch.topk(probabilities, k=top_k)
        
        predictions = []
        for idx, prob in zip(top_idxs.tolist(), top_probs.tolist()):
            predictions.append({
                "class_name": classes[idx],
                "confidence": float(prob)
            })

        predicted_idx = top_idxs[0].item()
        return {
            "status": "success",
            "predicted_class": classes[predicted_idx],
            "confidence": float(top_probs[0].item()),
            "predictions": predictions
        }

    except Exception as e:
        import traceback
        print(f"Error handling job: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

# Allow local testing
if __name__ == "__main__":
    if runpod is not None:
        runpod.serverless.start({"handler": handler})
    else:
        print("Runpod package not detected. Running local verification interface.")
        # Load model and print output
        try:
            load_model()
            print("Model loaded successfully. Ready to run tests.")
        except Exception as err:
            print(f"Failed to load model: {err}")
