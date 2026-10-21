import io
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# 1. Exact Architecture Specification for BRAHMA_CV50
class ProduceResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet50(weights=None)
        # 2048-dimensional bottleneck output from layer4 + avgpool
        self.conv_base = nn.Sequential(*list(base.children())[:-1])
        
        # Dense regularized head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(base.fc.in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_base(x)
        logits = self.classifier(features)
        return logits


# 2. Hardware Acceleration Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[BRAHMA_CV50] Bootstrapping runtime on device: {device}")

# 3. Load Checkpoint
WEIGHTS_PATH = Path("weights/BRAHMA_CV50_PRODUCTION.pt")
if not WEIGHTS_PATH.exists():
    # Fallback to local root check if weights/ folder is omitted
    WEIGHTS_PATH = Path("BRAHMA_CV50_PRODUCTION.pt")

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Could not locate BRAHMA_CV50_PRODUCTION.pt in './weights/' or current directory."
    )

print(f"[BRAHMA_CV50] Loading weights from {WEIGHTS_PATH}...")
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

class_names: List[str] = checkpoint.get("class_names", [])
num_classes: int = checkpoint.get("num_classes", len(class_names))

model = ProduceResNet50(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"[BRAHMA_CV50] Production model initialized with {num_classes} classes.")

# 4. Standard Preprocessing Pipeline
infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. FastAPI App & Template Engine
app = FastAPI(
    title="BRAHMA_CV50 Production Engine",
    description="Fine-grained Botanical Computer Vision Model by Namish Kumar",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "num_classes": num_classes,
            "device": str(device).upper()
        }
    )

@app.post("/predict")
async def predict_sample(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        raw_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image parsing failed: {str(e)}")

    # Tensor dispatch
    tensor_input = infer_transforms(pil_image).unsqueeze(0).to(device)

    # Inference with device synchronization for accurate wall-clock latency
    if device.type == "mps":
        torch.mps.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.softmax(logits, dim=1)[0]

    if device.type == "mps":
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - t_start) * 1000

    # Top-5 Ranking
    k = min(5, num_classes)
    top_probs, top_indices = torch.topk(probabilities, k=k)

    top_results: List[Dict[str, Any]] = []
    for prob, idx in zip(top_probs, top_indices):
        label = class_names[idx.item()] if idx.item() < len(class_names) else f"Class_{idx.item()}"
        top_results.append({
            "label": label.replace("_", " ").title(),
            "confidence": round(float(prob.item() * 100), 2)
        })

    return JSONResponse({
        "status": "success",
        "best_prediction": top_results[0]["label"],
        "confidence": top_results[0]["confidence"],
        "top5": top_results,
        "latency_ms": round(latency_ms, 2),
        "device": str(device).upper()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)