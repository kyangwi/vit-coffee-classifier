# Use PyTorch runtime image with GPU support (CUDA 12.1)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install RunPod, Hugging Face hub, timm, and other dependencies
RUN pip install --no-cache-dir \
    runpod==1.6.2 \
    timm==0.9.12 \
    pillow==10.1.0 \
    numpy==1.26.2 \
    huggingface_hub==0.19.4 \
    fastcore \
    fastai

# Set working directory inside the container
WORKDIR /app

# Copy serverless handler code
COPY rp_handler.py /app/rp_handler.py

# Bake model into the image:
# Copy local file if present, or download during docker build
COPY vit_base_patch16_224_coffee_preloaded.pth* /app/
RUN python -c "import os; from huggingface_hub import hf_hub_download; os.path.exists('/app/vit_base_patch16_224_coffee_preloaded.pth') or hf_hub_download(repo_id='Bwenge840/vit-base-patch16-224-coffee-preloaded', filename='vit_base_patch16_224_coffee_preloaded.pth', local_dir='/app')"

# Set environment variables for the baked model
ENV MODEL_PATH="/app/vit_base_patch16_224_coffee_preloaded.pth"
ENV HF_REPO_ID="Bwenge840/vit-base-patch16-224-coffee-preloaded"
ENV HF_FILENAME="vit_base_patch16_224_coffee_preloaded.pth"

# Command to execute the RunPod serverless entry point
CMD ["python", "-u", "/app/rp_handler.py"]
