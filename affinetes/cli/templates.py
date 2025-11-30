"""Environment template definitions for CLI init command"""

# Function-based environment with Actor class
ACTOR_ENV_PY = '''"""Environment implementation with Actor class

This is a simple calculator environment demonstrating affinetes usage.
"""

import os


class Actor:
    """Calculator actor for arithmetic operations"""
    
    def __init__(self):
        """Initialize actor with environment variables"""
        self.precision = int(os.getenv("PRECISION", "2"))
    
    async def add(self, a: float, b: float) -> dict:
        """
        Add two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Addition result
        """
        result = a + b
        return {
            "operation": "add",
            "a": a,
            "b": b,
            "result": round(result, self.precision)
        }
    
    async def multiply(self, a: float, b: float) -> dict:
        """
        Multiply two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Multiplication result
        """
        result = a * b
        return {
            "operation": "multiply",
            "a": a,
            "b": b,
            "result": round(result, self.precision)
        }
    
    async def batch_calculate(self, operations: list) -> dict:
        """
        Execute batch calculations
        
        Args:
            operations: List of operations, e.g., [{"op": "add", "a": 1, "b": 2}, ...]
            
        Returns:
            Batch calculation results
        """
        results = []
        for op_data in operations:
            op = op_data.get("op")
            a = op_data.get("a")
            b = op_data.get("b")
            
            if op == "add":
                result = await self.add(a, b)
            elif op == "multiply":
                result = await self.multiply(a, b)
            else:
                result = {"error": f"Unknown operation: {op}"}
            
            results.append(result)
        
        return {
            "total": len(operations),
            "results": results
        }
'''

# Function-based environment with module-level functions
BASIC_ENV_PY = '''"""Environment implementation with module-level functions

This is a simple calculator environment demonstrating affinetes usage.
"""

import os


async def add(a: float, b: float) -> dict:
    """
    Add two numbers
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Addition result
    """
    precision = int(os.getenv("PRECISION", "2"))
    result = a + b
    
    return {
        "operation": "add",
        "a": a,
        "b": b,
        "result": round(result, precision)
    }


async def multiply(a: float, b: float) -> dict:
    """
    Multiply two numbers
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Multiplication result
    """
    precision = int(os.getenv("PRECISION", "2"))
    result = a * b
    
    return {
        "operation": "multiply",
        "a": a,
        "b": b,
        "result": round(result, precision)
    }


async def batch_calculate(operations: list) -> dict:
    """
    Execute batch calculations
    
    Args:
        operations: List of operations, e.g., [{"op": "add", "a": 1, "b": 2}, ...]
        
    Returns:
        Batch calculation results
    """
    results = []
    for op_data in operations:
        op = op_data.get("op")
        a = op_data.get("a")
        b = op_data.get("b")
        
        if op == "add":
            result = await add(a, b)
        elif op == "multiply":
            result = await multiply(a, b)
        else:
            result = {"error": f"Unknown operation: {op}"}
        
        results.append(result)
    
    return {
        "total": len(operations),
        "results": results
    }
'''

# Function-based Dockerfile (HTTP server auto-injected)
FUNCTION_DOCKERFILE = '''FROM python:3.12-slim

WORKDIR /app

# Copy environment code
COPY . /app/

# Note: HTTP server will be auto-injected by affinetes
'''

# HTTP-based environment with FastAPI
FASTAPI_ENV_PY = '''"""Environment implementation with FastAPI"""

import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Custom Environment API")


class ProcessRequest(BaseModel):
    data: dict


class EvaluateRequest(BaseModel):
    task: str
    params: dict = {}


@app.get("/health")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Environment is running"}


@app.post("/process")
async def process(request: ProcessRequest):
    """Process data endpoint"""
    api_key = os.getenv("API_KEY")
    
    return {
        "status": "success",
        "input": request.data,
        "api_key_set": bool(api_key),
        "message": "Processed by FastAPI"
    }


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Evaluate task endpoint"""
    return {
        "task": request.task,
        "score": 1.0,
        "success": True,
        "params": request.params
    }
'''

# HTTP-based Dockerfile
FASTAPI_DOCKERFILE = '''FROM python:3.12-slim

WORKDIR /app

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy environment code
COPY . /app/

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000"]
'''