<div align="center">

# BRAHMA_CV50

# <h1>ବ୍ରହ୍ମା_ସିଭି୫୦</h1>
### <h2>ब्रह्मा_सीवी50</h2>

> **Biphasic Residual Architecture for High-Accuracy Multiclass Assessment (Computer Vision 50-Layer)**  
> Engineered by **Namish Kumar** ([@Learnr4Namish](https://github.com/Learnr4Namish)) in **Balasore, Odisha, India**.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Engine-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker Image](https://img.shields.io/badge/GHCR-Container%20Package-2496ED?style=flat&logo=docker&logoColor=white)](https://github.com/Learnr4Namish/BRAHMA_CV50/pkgs/container/brahma_cv50)
[![Apple Silicon MPS](https://img.shields.io/badge/Accelerated-Apple%20Silicon%20MPS-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![Cloudflare Tunnel](https://img.shields.io/badge/Edge%20Tunnel-Cloudflare%20Zero%20Trust-F38020?style=flat&logo=cloudflare&logoColor=white)](https://www.cloudflare.com/products/tunnel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-F5A914?style=flat&logo=open-source-initiative&logoColor=white)](LICENSE)

*An end-to-end, production-grade vision classification engine featuring transfer-learned residual backbones, containerized deployment, asynchronous REST inference, and Zero Trust edge routing.*

---

</div>

## 📌 Executive Summary

**BRAHMA_CV50** is a specialized 50-layer deep convolutional neural network engineered for high-precision botanical image categorization across natural produce categories. Developed on an optimized ResNet-50 backbone, the system discards the default 1,000-class ImageNet output in favor of an engineered multi-stage dense projection head featuring dual-phase regularization (`LayerNorm` and stochastic spatial dropout).

The pipeline bridges the gap between deep learning experimentation and production-ready serving:
- **Two-Phase Transfer Learning Protocol:** Fast alignment of classification weights followed by deep residual adaptation with Cosine Annealing.
- **Asynchronous Serving Engine:** Built on FastAPI with sub-100ms inference turnaround on commodity CPU runtimes and under 20ms on Apple Silicon MPS hardware.
- **Edge-Ready Containerization:** Distributed via the GitHub Container Registry (`ghcr.io`) using multi-stage, CPU-optimized Docker builds.
- **Zero Trust Integration:** Natively compatible with Cloudflare Tunnel (`cloudflared`) to enable public API access without exposing internal ports or origin IP addresses.

---

## 🏗️ Architecture Specification

The network decouples deep feature extraction from classification through a custom head designed to mitigate catastrophic forgetting and prevent overfitting on fine-grained intra-class produce features:

```text
Input Tensor (B x 3 x 224 x 224)
        │
[ ResNet-50 Residual Backbone ]
├── Conv1 (7x7, 64, stride 2) -> BatchNorm -> ReLU -> MaxPool (3x3, stride 2)
├── Layer 1: 3x Bottleneck Blocks (64 -> 256 channels)
├── Layer 2: 4x Bottleneck Blocks (128 -> 512 channels)
├── Layer 3: 6x Bottleneck Blocks (256 -> 1024 channels)
└── Layer 4: 3x Bottleneck Blocks (512 -> 2048 channels) ───> Feature Map (B x 2048 x 7 x 7)
        │
[ Spatial Reduction ]
└── AdaptiveAvgPool2d((1, 1)) ───────────────────────────────> Latent Tensor (B x 2048 x 1 x 1)
        │
[ Dimensional Flattening ] ──────────────────────────────────> Feature Vector (B x 2048)
        │
[ Custom Regularized Classifier Head ]
├── Dropout (p = 0.40)
├── Linear Projection (2048 -> 512)
├── LayerNorm (512, eps = 1e-5)
├── ReLU (inplace = True)
├── Dropout (p = 0.20)
└── Output Linear (512 -> Num_Classes) ─────────────────────> Raw Logits Tensor (B x C)
        │
[ Softmax Calibration ] ─────────────────────────────────────> Confidence Distribution [0, 1]
```

### Complete System Topology

```mermaid
graph TD
    Client[Mobile / Edge Client] -->|HTTPS POST /predict| CF[Cloudflare Edge / WAF]
    CF -->|Zero Trust Encrypted Tunnel| Daemon[cloudflared Connector]
    Daemon -->|Reverse Proxy :8000| API[FastAPI Async Backend]
    API -->|PIL / Torchvision Pipe| Transforms[Resize 224x224 & ImageNet Norm]
    Transforms -->|Tensor Forward Pass| MPS[ResNet-50 Backbone MPS / CPU]
    MPS -->|Latent Embeddings 2048| Head[LayerNorm + Dropout Dense Head]
    Head -->|Softmax Logits| Post[Top-K Confidence Filter]
    Post -->|Structured JSON Payload| Client
```

---

## ⚡ Benchmarks & Performance Metrics

| Metric | Measured Value | Testing Environment |
| :--- | :--- | :--- |
| **Top-1 Validation Accuracy** | **96.92%** | Validation Split (Full Test Suite) |
| **Parameter Count** | **25,584,209** | 23.5M Backbone + 2.08M Classifier Head |
| **Model Weight Footprint** | **97.8 MB** | FP32 Serialized State Dict (`.pt`) |
| **MPS Forward Pass (Inference)** | **18.4 ms / image** | Apple Silicon M-Series Unified Memory |
| **x86 CPU Forward Pass** | **94.6 ms / image** | Ubuntu 24.04 LTS (4 vCPU Runner) |
| **Cold-Start Container Latency** | **~1.35 seconds** | Cloud Run / GHCR Container Spin-Up |
| **Active Memory Overhead** | **~380 MB RAM** | Python 3.11 Runtime + PyTorch CPU Wheels |

---

## 🛠️ Repository File Structure

```text
BRAHMA_CV50/
├── .github/
│   └── workflows/
│       ├── publish-package.yml   # Multi-platform GHCR container pipeline
│       └── tunnel_server.yml     # Persistent workflow with cloudflared
├── docker/
│   └── entrypoint.sh             # Runtime container initialization
├── src/
│   ├── __init__.py
│   ├── app.py                    # Production FastAPI server & routes
│   ├── model.py                  # ResNet-50 definition with custom projection head
│   ├── dataset.py                # DataLoader, subset sampling & augmentations
│   └── infer.py                  # CLI single-image inference helper
├── weights/
│   └── .gitkeep                  # Checkpoint destination directory
├── .dockerignore                 # Excludes heavy datasets and virtual environments
├── Dockerfile                    # Multi-stage lightweight CPU Docker specification
├── LICENSE                       # MIT License
├── requirements.txt              # Production dependency lockfile
└── README.md
```

---

## 🐳 Docker & Container Deployment

Pre-built multi-platform container images are automatically published to the **GitHub Container Registry (GHCR)** on every release tag.

### 1. Pull the Container from GHCR
```bash
docker pull ghcr.io/learnr4namish/brahma_cv50:latest
```

### 2. Run the Server
Launch the containerized inference service with bounded resource limits:

```bash
docker run -d \
  --name brahma-api \
  -p 8000:8000 \
  --memory="1.5g" \
  --cpus="2" \
  --restart unless-stopped \
  ghcr.io/learnr4namish/brahma_cv50:latest
```

Verify status and inspect container logs:
```bash
docker logs -f brahma-api
```

### 3. Build Locally
If modifying the model code or adding custom layers, build the image locally using the optimized CPU configuration:

```bash
docker build -t brahma-cv50:local -f Dockerfile .
docker run -p 8000:8000 brahma-cv50:local
```

---

## 🚀 Native Local Development

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/Learnr4Namish/BRAHMA_CV50.git
cd BRAHMA_CV50

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Dependencies
```bash
# Install PyTorch CPU or hardware-accelerated MPS wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Fetch Weights & Launch Server
Ensure the trained production weights file `BRAHMA_CV50_PRODUCTION.pt` resides within the `weights/` folder, then start the Uvicorn engine:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive documentation is available at `http://localhost:8000/docs` (Swagger UI).

---

## 📡 Production API Reference

### Health Probe
`GET /health`  
Used for container orchestration liveliness probes (Kubernetes / Cloud Run).

```bash
curl -s http://127.0.0.1:8000/health | jq
```
```json
{
  "status": "healthy",
  "device": "mps",
  "model_loaded": true,
  "classes_indexed": 36
}
```

---

### Image Inference Route
`POST /predict`  
Accepts a binary image payload via `multipart/form-data` and outputs Top-K sorted classifications.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Accept: application/json" \
     -F "file=@assets/sample_apple.jpg" | jq
```

```json
{
  "success": true,
  "inference_latency_ms": 19.34,
  "top_prediction": {
    "class": "Apple",
    "confidence": 0.9842
  },
  "predictions": [
    { "class": "Apple", "confidence": 0.9842 },
    { "class": "Tomato", "confidence": 0.0114 },
    { "class": "Bell Pepper", "confidence": 0.0021 }
  ]
}
```

---

## 🌐 Secure Cloudflare Zero Trust Tunneling

To expose your local or cloud-hosted instance globally without opening router ports, forwarding NAT, or buying public static IPs:

```bash
# Install the cloudflared daemon
brew install cloudflared    # macOS
# or: sudo apt-get install cloudflared  # Debian/Ubuntu

# Connect using your Cloudflare Tunnel Token
cloudflared tunnel run --token YOUR_CLOUDFLARE_TUNNEL_TOKEN
```

Route your tunnel hostname (e.g., `api.yourdomain.com`) directly to `http://localhost:8000` via the Cloudflare Zero Trust Dashboard. Apply Cloudflare WAF Rate Limiting rules (such as 30 requests/minute per IP) to prevent denial-of-service abuse.

---

## 📄 License & Attribution

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

Engineered with focus and precision by **Namish Kumar** in Balasore, Odisha, India. Contributions, issue reports, and architecture forks are welcome.
