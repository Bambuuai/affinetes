# Targon-SDK Example: LLM Deployment on Targon

Deploy LLM service on Targon GPU with automatic port exposure.

## Usage

### Deploy LLM Server

```bash
# Deploy with default model (Qwen/Qwen2.5-7B-Instruct)
targon deploy affinetes/examples/targon-sdk/targon_llm_server.py

# Deploy with custom model
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
MODEL_REVISION="main" \
targon deploy affinetes/examples/targon-sdk/targon_llm_server.py

# Get llm url
targon app get app-xxx 
```

The deployment will expose an HTTPS URL with OpenAI-compatible API.

### Run Evaluation

```bash
# Run evaluation with deployed LLM
python examples/targon-sdk/local_evaluator.py \
    --llm-url "https://fnc-xxxx.serverless.targon.com/v1" \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --image "docker.io/affinefoundation/lgc:pi" \
    --task-id-start 1 \
    --task-id-end 2
```