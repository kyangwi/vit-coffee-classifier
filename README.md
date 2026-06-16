# Deploying the ViT Model on RunPod Serverless

This guide walks you through building, pushing, and deploying the Vision Transformer (ViT) coffee classifier model to RunPod Serverless.

---

## Prerequisites

1. **Docker**: Ensure Docker is installed and running on your development machine.
2. **Docker Hub (or other registry) Account**: A registry where you can host the container image.
3. **RunPod Account**: A RunPod account with billing enabled.

---

## Step 1: Prepare the Files

Ensure you have the following files in your folder:
- `Dockerfile`
- `rp_handler.py`

---

## Step 2: Build the Docker Image

Open a terminal in the `runpod_deployment/` directory and build the Docker image. Replace `username` with your Docker Hub username.

```bash
docker build -t username/vit-coffee-classifier:latest .
```

*Note: The model weights will download at container startup from the Hugging Face hub by default (`Bwenge840/vit-base-patch16-224-coffee-preloaded`).*

---

## Step 3: Push the Image to Docker Hub

Log in to Docker Hub and push your newly built image.

```bash
docker login
docker push username/vit-coffee-classifier:latest
```

---

## Step 4: Create a RunPod Serverless Endpoint

1. Go to the [RunPod Console](https://www.runpod.io/console/serverless) and log in.
2. Click on **Serverless** -> **Endpoints** -> **New Endpoint**.
3. Fill in the following details:
   - **Endpoint Name**: `vit-coffee-classifier`
   - **Container Image**: `username/vit-coffee-classifier:latest` (use your Docker Hub image tag)
   - **Container Registry**: Keep empty if public (or choose credentials if private)
   - **Active GPU Types**: Select a GPU like `NVIDIA RTX 4090` or `NVIDIA A10G` (minimum 8GB VRAM is plenty).
   - **Min Provisioned Workers**: `0` (this ensures it scales down to 0 when idle to avoid charges).
   - **Max Workers**: `3` (adjust based on concurrency requirements).
   - **Idle Timeout**: `300` seconds (runs keep-alive to avoid cold starts on successive runs).
4. Click **Create**.
5. Save your **Endpoint ID** (displayed next to the endpoint name) and your **RunPod API Key** (found under API Keys in user settings).

---

## Step 5: Configure the Django Application

Use these credentials to configure your Django settings. In your `.env` file or terminal environment variables, set:
```env
RUNPOD_API_KEY="your_runpod_api_key_here"
RUNPOD_ENDPOINT_ID="your_endpoint_id_here"
```
