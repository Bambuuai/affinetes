#!/usr/bin/env python3
"""
Targon LLM Server - Deploy LLM service on Targon GPU

No Docker-in-Docker required, no privileged mode needed.

Usage:
    # Deploy with environment variables:
    MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" MODEL_REVISION="main" \
        targon deploy examples/targon-sdk/targon_llm_server.py
    
    # Or edit MODEL_NAME and MODEL_REVISION in the file, then:
    targon deploy examples/targon-sdk/targon_llm_server.py
"""

import logging
import os
import targon

logger = logging.getLogger(__name__)

# Build image with vLLM
vllm_image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .pip_install(
        "vllm==0.10.2",
        "torch==2.8.0",
        "huggingface_hub==0.35.0",
        "hf_transfer"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        "MODEL_REVISION": os.environ.get("MODEL_REVISION", "main"),
    })
)

app = targon.App("affine-llm", image=vllm_image)


@app.function(
    resource=targon.Compute.H200_SMALL,
    max_replicas=1,
)
@targon.web_server(port=8000)
def serve_llm():
    """
    Deploy LLM service and expose OpenAI-compatible API
    
    GPU detection is automatic - uses all available GPUs for tensor parallelism.
    Model configuration is read from environment variables.
    """
    import subprocess
    import torch
    import os
    
    # Read configuration from environment variables
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    model_revision = os.environ.get("MODEL_REVISION", "main")
    
    # Auto-detect number of GPUs
    n_gpu = torch.cuda.device_count()
    
    logger.info(f"Starting vLLM service with model: {model_name}")
    logger.info(f"  Revision: {model_revision}")
    logger.info(f"  Detected GPUs: {n_gpu}")
    
    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_name,
        "--revision",
        model_revision,
        "--served-model-name",
        model_name,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        str(n_gpu),
    ]
    
    logger.info(f"Launching vLLM: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)