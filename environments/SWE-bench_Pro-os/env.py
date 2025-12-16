"""SWE-bench Pro evaluation environment using mini-swe-agent"""
import os
import time
import tempfile
import subprocess
import yaml
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Actor:
    """SWE-bench Pro environment actor"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize actor and load dataset"""
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        
        # Load SWE-bench Verified dataset
        print("Loading SWE-bench Verified dataset...")
        dataset = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
        
        # Create task_id -> instance mapping (sorted by instance_id for consistency)
        sorted_instances = sorted(dataset, key=lambda x: x["instance_id"])
        self.instances = {idx: inst for idx, inst in enumerate(sorted_instances)}
        
        print(f"Loaded {len(self.instances)} instances")
    
    def _get_swebench_image_name(self, instance: dict) -> str:
        """Get SWE-bench Docker image name for an instance"""
        iid = instance["instance_id"]
        # Docker doesn't allow double underscore, replace with magic token
        id_docker = iid.replace("__", "_1776_")
        return f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
    
    def _verify_patch(self, instance: dict, patch: str) -> float:
        """Verify patch by running tests in evaluation container
        
        Returns:
            1.0 if patch passes all required tests, 0.0 otherwise
        """
        if not patch or not patch.strip():
            return 0.0
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write patch to file
                patch_file = os.path.join(tmpdir, "patch.diff")
                with open(patch_file, "w") as f:
                    f.write(patch)
                
                # Get evaluation image
                image = self._get_swebench_image_name(instance)
                
                # Build test command
                base_commit = instance.get("base_commit", "")
                test_cmd = f"""
                cd /app
                git reset --hard {base_commit}
                git checkout {base_commit}
                git apply -v /workspace/patch.diff || exit 1
                """
                
                # Run container with patch
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{tmpdir}:/workspace",
                    image,
                    "/bin/bash", "-c", test_cmd
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=600,
                    text=True
                )
                
                # Check if patch applied successfully and tests passed
                if result.returncode == 0:
                    return 1.0
                else:
                    return 0.0
                    
        except Exception as e:
            print(f"Error verifying patch: {e}")
            return 0.0
    
    async def evaluate(
        self,
        task_id: int,
        model: str = "deepseek-ai/DeepSeek-V3",
        base_url: str = "https://llm.chutes.ai/v1",
        api_key: Optional[str] = None,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
    ):
        """Evaluate model on a SWE-bench Pro task
        
        Args:
            task_id: Numeric task ID (index into dataset)
            model: Model name for LiteLLM
            base_url: API base URL
            api_key: API key (falls back to self.api_key)
            timeout: Timeout for each command execution
            temperature: Model temperature
            seed: Random seed
            max_iterations: Maximum agent steps
            cost_limit: Maximum cost in dollars
        
        Returns:
            Result dict with score, patch, and conversation
        """
        start = time.time()
        current_api_key = api_key or self.api_key
        
        # Validate task_id
        if task_id not in self.instances:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-{len(self.instances)-1}")
        
        instance = self.instances[task_id]
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        
        print(f"Evaluating instance: {instance_id}")
        
        # Configure model
        # For custom OpenAI-compatible endpoints, LiteLLM requires "openai/" prefix
        # This tells LiteLLM to use OpenAI SDK with custom api_base
        litellm_model = f"openai/{model}" if not model.startswith("openai/") else model
        model_obj = LitellmModel(
            model_name=litellm_model,
            model_kwargs={
                "api_base": base_url,  # LiteLLM uses api_base, not base_url
                "api_key": current_api_key,
                "temperature": temperature,
            },
            cost_tracking="ignore_errors",  # Ignore cost calculation errors for custom models
        )
        
        # Configure environment (SWE-bench Docker)
        # Use unique container name to avoid conflicts in concurrent execution
        container_name = f"swebench-{task_id}-{int(time.time() * 1000)}"
        env = DockerEnvironment(
            image=self._get_swebench_image_name(instance),
            timeout=timeout,
            executable="docker",  # DOOD: use host docker
            run_args=["--rm", "--name", container_name],  # Ensure cleanup and unique naming
            container_timeout=str(timeout),
        )
        
        # Load SWE-bench agent configuration from YAML (in Docker image)
        config_path = Path(__file__).parent / "swebench_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override agent config limits with function parameters
        agent_config = config["agent"].copy()
        agent_config["step_limit"] = max_iterations
        agent_config["cost_limit"] = cost_limit
        
        # Create and run agent with proper SWE-bench configuration
        agent = DefaultAgent(
            model_obj,
            env,
            **agent_config,
        )
        
        exit_status = "unknown"
        result = ""
        patch = ""
        error = None
        
        try:
            # Run agent.run() in thread pool to avoid blocking event loop
            # This allows multiple concurrent evaluate() calls to run in parallel
            loop = asyncio.get_event_loop()
            exit_status, result = await loop.run_in_executor(
                None,
                agent.run,
                problem_statement
            )
            patch = result  # Final output is the patch
            
        except Exception as e:
            import traceback
            exit_status = type(e).__name__
            result = str(e)
            patch = ""
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"Error running agent: {e}")
        
        finally:
            # Clean up environment
            try:
                env.cleanup()
            except Exception as cleanup_error:
                try:
                    if hasattr(env, 'container_id') and env.container_id:
                        subprocess.run(
                            ["docker", "rm", "-f", env.container_id],
                            capture_output=True,
                            timeout=30,
                        )
                        print(f"Force removed container {env.container_id}")
                except Exception as force_cleanup_error:
                    print(f"Error: Force cleanup also failed for task {task_id}: {force_cleanup_error}")
        
        # Verify patch
        score = self._verify_patch(instance, patch) if patch else 0.0
        
        # Extract usage information and clean conversation
        # Collect usage from all messages that have it in their extra field
        # Then remove extra field from conversation
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_tokens = 0
        clean_conversation = []
        
        for msg in agent.messages:
            if isinstance(msg, dict) and "extra" in msg:
                msg_extra = msg.get("extra", {})
                if isinstance(msg_extra, dict):
                    msg_usage = None
                    # Check for response.usage pattern
                    if "response" in msg_extra and isinstance(msg_extra["response"], dict):
                        msg_usage = msg_extra["response"].get("usage")
                    # Check for direct usage pattern
                    elif "usage" in msg_extra:
                        msg_usage = msg_extra["usage"]
                    
                    # Accumulate usage tokens
                    if msg_usage and isinstance(msg_usage, dict):
                        total_completion_tokens += msg_usage.get("completion_tokens", 0)
                        total_prompt_tokens += msg_usage.get("prompt_tokens", 0)
                        total_tokens += msg_usage.get("total_tokens", 0)
                
                # Remove extra field from message
                clean_msg = {k: v for k, v in msg.items() if k != "extra"}
                clean_conversation.append(clean_msg)
            else:
                clean_conversation.append(msg)
        
        # Create aggregated usage object
        usage = {
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens
        } if total_tokens > 0 else None
        
        # Return result
        result_dict = {
            "task_name": "swe-bench-pro",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "instance_id": instance_id,
                "patch": patch,
                "conversation": clean_conversation,  # Cleaned conversation without extra field
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "task_id": task_id,
                "usage": usage,  # Aggregated usage info with only token counts
            }
        }
        
        # Add error info if present
        if error:
            result_dict["extra"]["error"] = error
            result_dict["extra"]["error_type"] = exit_status
        
        return result_dict
    
    async def local_evaluate(
        self,
        task_id: int,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        timeout: int = 1800,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_iterations: int = 30,
        cost_limit: float = 10.0,
        max_tokens: int = 2048,
        input_cost_per_million: float = 0.01,
        output_cost_per_million: float = 0.03,
    ):
        """Evaluate model on a SWE-bench Pro task using local transformers model
        
        Args:
            task_id: Numeric task ID (index into dataset)
            model: Loaded AutoModelForCausalLM instance
            tokenizer: Loaded AutoTokenizer instance
            timeout: Timeout for each command execution
            temperature: Model temperature
            seed: Random seed
            max_iterations: Maximum agent steps
            cost_limit: Maximum cost in dollars
            max_tokens: Maximum tokens to generate per response
            input_cost_per_million: Cost per million input tokens (default: $0.01)
            output_cost_per_million: Cost per million output tokens (default: $0.03)
        
        Returns:
            Result dict with score, patch, and conversation
        """
        start = time.time()
        
        # Validate task_id
        if task_id not in self.instances:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0-{len(self.instances)-1}")
        
        instance = self.instances[task_id]
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        
        print(f"Evaluating instance: {instance_id}")
        
        # Create a custom model wrapper that matches LitellmModel interface
        class TransformersModel:
            """Local transformers model wrapper matching mini-swe-agent Model interface"""
            
            def __init__(self, model, tokenizer, temperature=0.0, seed=None, max_tokens=2048,
                        input_cost_per_million=0.01, output_cost_per_million=0.03):
                self.model = model
                self.tokenizer = tokenizer
                self.temperature = temperature
                self.seed = seed
                self.max_tokens = max_tokens
                self.cost = 0.0
                self.n_calls = 0
                self.input_cost_per_million = input_cost_per_million
                self.output_cost_per_million = output_cost_per_million
                
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
                """Query the local model. Must match LitellmModel.query() signature.
                
                Returns dict with 'content' and 'extra' keys matching LitellmModel format.
                """
                self.n_calls += 1
                
                # Extract only role and content from messages (ignore timestamp, extra, etc.)
                clean_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
                
                # Convert messages to model input using chat template
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    prompt = self.tokenizer.apply_chat_template(
                        clean_messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    # Fallback: simple concatenation for models without chat template
                    prompt = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in clean_messages])
                    prompt += "\n\nASSISTANT:"
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length or 4096
                ).to(self.model.device)
                
                # Set seed for reproducibility
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(self.seed)
                
                # Prepare generation kwargs
                gen_kwargs = {
                    "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                # Add temperature/sampling settings
                if self.temperature > 0:
                    gen_kwargs.update({
                        "temperature": self.temperature,
                        "do_sample": True,
                        "top_p": kwargs.get("top_p", 0.95),
                    })
                else:
                    gen_kwargs["do_sample"] = False
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode only the new tokens (exclude the prompt)
                prompt_length = inputs['input_ids'].shape[1]
                response_ids = outputs[0][prompt_length:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Calculate token counts
                completion_tokens = len(response_ids)
                prompt_tokens = prompt_length
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost based on token usage
                input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_million
                output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_million
                call_cost = input_cost + output_cost
                self.cost += call_cost
                
                # Return in the format expected by DefaultAgent
                # Must have 'content' key and 'extra' dict with usage info
                return {
                    "content": response_text,
                    "extra": {
                        "response": {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": response_text
                                }
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }
                    }
                }
            
            def get_template_vars(self) -> dict:
                """Return template variables for rendering prompts.
                Must match LitellmModel.get_template_vars() signature.
                """
                return {
                    "n_model_calls": self.n_calls,
                    "model_cost": self.cost,
                    "model_name": self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else "local",
                }
        
        # Create model wrapper
        model_obj = TransformersModel(
            model, 
            tokenizer, 
            temperature=temperature, 
            seed=seed,
            max_tokens=max_tokens,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million
        )
        
        # Configure environment (SWE-bench Docker)
        container_name = f"swebench-{task_id}-{int(time.time() * 1000)}"
        env = DockerEnvironment(
            image=self._get_swebench_image_name(instance),
            timeout=timeout,
            executable="docker",
            run_args=["--rm", "--name", container_name],
            container_timeout=str(timeout),
        )
        
        # Load SWE-bench agent configuration
        config_path = Path(__file__).parent / "swebench_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Override agent config limits
        agent_config = config["agent"].copy()
        agent_config["step_limit"] = max_iterations
        agent_config["cost_limit"] = cost_limit
        
        # Create and run agent
        agent = DefaultAgent(model_obj, env, **agent_config)
        
        exit_status = "unknown"
        result = ""
        patch = ""
        error = None
        
        try:
            # Run agent.run() in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            exit_status, result = await loop.run_in_executor(
                None,
                agent.run,
                problem_statement
            )
            patch = result
            
        except Exception as e:
            import traceback
            exit_status = type(e).__name__
            result = str(e)
            patch = ""
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            print(f"Error running agent: {e}")
        
        finally:
            # Clean up environment
            try:
                env.cleanup()
            except Exception:
                try:
                    if hasattr(env, 'container_id') and env.container_id:
                        subprocess.run(
                            ["docker", "rm", "-f", env.container_id],
                            capture_output=True,
                            timeout=30,
                        )
                        print(f"Force removed container {env.container_id}")
                except Exception as force_cleanup_error:
                    print(f"Error: Force cleanup failed for task {task_id}: {force_cleanup_error}")
        
        # Verify patch
        score = self._verify_patch(instance, patch) if patch else 0.0
        
        # Extract usage information and clean conversation
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_tokens = 0
        clean_conversation = []
        
        for msg in agent.messages:
            if isinstance(msg, dict) and "extra" in msg:
                msg_extra = msg.get("extra", {})
                if isinstance(msg_extra, dict):
                    msg_usage = None
                    if "response" in msg_extra and isinstance(msg_extra["response"], dict):
                        msg_usage = msg_extra["response"].get("usage")
                    elif "usage" in msg_extra:
                        msg_usage = msg_extra["usage"]
                    
                    if msg_usage and isinstance(msg_usage, dict):
                        total_completion_tokens += msg_usage.get("completion_tokens", 0)
                        total_prompt_tokens += msg_usage.get("prompt_tokens", 0)
                        total_tokens += msg_usage.get("total_tokens", 0)
                
                clean_msg = {k: v for k, v in msg.items() if k != "extra"}
                clean_conversation.append(clean_msg)
            else:
                clean_conversation.append(msg)
        
        usage = {
            "completion_tokens": total_completion_tokens,
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_tokens
        } if total_tokens > 0 else None
        
        # Return result
        result_dict = {
            "task_name": "swe-bench-pro",
            "score": score,
            "success": score > 0.0,
            "time_taken": time.time() - start,
            "extra": {
                "instance_id": instance_id,
                "patch": patch,
                "conversation": clean_conversation,
                "model_calls": agent.model.n_calls,
                "model_cost": agent.model.cost,
                "task_id": task_id,
                "usage": usage,
            }
        }
        
        if error:
            result_dict["extra"]["error"] = error
            result_dict["extra"]["error_type"] = exit_status
        
        return result_dict