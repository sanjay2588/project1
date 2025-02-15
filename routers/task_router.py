from fastapi import APIRouter, HTTPException, Query
from models.task import RunTaskRequest
from core.tasks import execute_task
from urllib.parse import unquote

router = APIRouter()

@router.post("/run")
async def run_task(task: str = Query(...)) -> dict:
    """Runs a specified task from query parameters."""
    try:
        decoded_task = unquote(task)
        request = RunTaskRequest(task=decoded_task)
        return await execute_task(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/read")
async def read_file(path: str = Query(...)) -> str:
    """Reads a file from the data directory."""
    try:
        from pathlib import Path
        from core.security import secure_path
        
        decoded_path = unquote(path)
        file_path = secure_path(Path("data") / decoded_path.lstrip("/"))
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {decoded_path}")
            
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 