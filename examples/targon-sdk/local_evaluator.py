#!/usr/bin/env python3
"""
Local Evaluator - Run affinetes evaluation locally with remote Targon LLM

This script runs affinetes evaluation framework locally and connects to
a remote LLM service deployed on Targon GPU.

No Docker-in-Docker required, no privileged mode needed.

Usage:
    python examples/targon-sdk/local_evaluator.py \
        --llm-url "https://your-targon-service.com/v1" \
        --model-name "Qwen/Qwen2.5-7B-Instruct" \
        --image "docker.io/affinefoundation/mth:pi" \
        --task-id-start 1 \
        --task-id-end 10
"""

import asyncio
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import affinetes as af


async def evaluate_task(env, task_id: int, model_name: str, llm_url: str, timeout: int) -> Dict[str, Any]:
    """Evaluate a single task"""
    try:
        print(f"[Task {task_id}] Starting evaluation...")
        start = time.time()
        
        result = await env.evaluate(
            model=model_name,
            base_url=llm_url,
            task_id=task_id,
            timeout=timeout,
            _timeout=timeout + 60,
            api_key="x",
        )
        
        elapsed = time.time() - start
        print(f"[Task {task_id}] Completed in {elapsed:.2f}s - Score: {result.get('score', 0)}")
        
        return {
            "task_id": task_id,
            "result": result,
            "elapsed": elapsed
        }
    except Exception as e:
        print(f"[Task {task_id}] ERROR: {type(e).__name__}: {str(e)}")
        return {
            "task_id": task_id,
            "error": str(e),
            "error_type": type(e).__name__,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Run affinetes evaluation locally with remote Targon LLM"
    )
    parser.add_argument(
        "--llm-url",
        required=True,
        help="URL of the Targon LLM service (e.g., https://your-service.com/v1)"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--image",
        default="docker.io/affinefoundation/mth:pi",
        help="Affinetes environment image (default: docker.io/affinefoundation/mth:pi)"
    )
    parser.add_argument(
        "--task-id-start",
        type=int,
        default=1,
        help="Start of task ID range (default: 1)"
    )
    parser.add_argument(
        "--task-id-end",
        type=int,
        default=10,
        help="End of task ID range (default: 10)"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        help="Comma-separated task IDs (e.g., 1,5,10,15,20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per task in seconds (default: 1800)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent evaluations (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Determine task IDs
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    else:
        task_ids = list(range(args.task_id_start, args.task_id_end + 1))
    
    print("=" * 80)
    print("LOCAL EVALUATOR - TARGON SDK MODE")
    print("=" * 80)
    print(f"LLM Service URL: {args.llm_url}")
    print(f"Model: {args.model_name}")
    print(f"Environment Image: {args.image}")
    print(f"Task IDs: {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}")
    print(f"Total Tasks: {len(task_ids)}")
    print(f"Concurrent: {args.concurrent}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 80)
    print()
    
    # Load affinetes environment locally
    print("Loading affinetes environment...")
    env = af.load_env(
        image=args.image,
        mode="docker",
        pull=True,
    )
    print(f"Environment loaded successfully")
    print()
    
    # Run evaluations
    start_time = time.time()
    
    if args.concurrent == 1:
        # Sequential evaluation
        results = []
        for task_id in task_ids:
            result = await evaluate_task(
                env=env,
                task_id=task_id,
                model_name=args.model_name,
                llm_url=args.llm_url,
                timeout=args.timeout,
            )
            results.append(result)
    else:
        # Concurrent evaluation
        print(f"Starting {len(task_ids)} evaluations with concurrency={args.concurrent}...")
        print()
        
        # Split tasks into batches
        batches = [task_ids[i:i + args.concurrent] for i in range(0, len(task_ids), args.concurrent)]
        results = []
        
        for i, batch in enumerate(batches):
            print(f"Batch {i+1}/{len(batches)}: Processing tasks {batch}")
            tasks = [
                evaluate_task(
                    env=env,
                    task_id=task_id,
                    model_name=args.model_name,
                    llm_url=args.llm_url,
                    timeout=args.timeout,
                )
                for task_id in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
    
    total_time = time.time() - start_time
    
    # Calculate summary statistics
    successful_tasks = [r for r in results if "result" in r]
    failed_tasks = [r for r in results if "error" in r]
    total_score = sum(r["result"].get("score", 0) for r in successful_tasks)
    avg_score = total_score / len(successful_tasks) if successful_tasks else 0
    
    # Print summary
    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Tasks: {len(task_ids)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    print(f"Average Score: {avg_score:.4f}")
    print(f"Total Score: {total_score:.2f}")
    print()
    
    # Print detailed results
    print("TASK DETAILS:")
    print("-" * 80)
    for result in results:
        if "error" in result:
            print(f"Task {result['task_id']}: ERROR - {result['error_type']}: {result['error']}")
        else:
            score = result['result'].get('score', 0)
            elapsed = result['elapsed']
            print(f"Task {result['task_id']}: SUCCESS - Score: {score:.4f}, Time: {elapsed:.2f}s")
    print("=" * 80)
    
    # Save results to file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(f"evaluation_results_{timestamp}.json")
    
    output_data = {
        "llm_url": args.llm_url,
        "model_name": args.model_name,
        "environment_image": args.image,
        "total_time": total_time,
        "total_tasks": len(task_ids),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "average_score": avg_score,
        "total_score": total_score,
        "task_ids": task_ids,
        "results": results,
    }
    
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())