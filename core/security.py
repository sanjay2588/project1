from pathlib import Path
from fastapi import HTTPException
import os

def validate_path_security(path: Path) -> bool:
    """Ensure path is within /data directory."""
    try:
        data_dir = Path("/data").resolve()
        target_path = path.resolve()
        return data_dir in target_path.parents
    except Exception:
        return False

def secure_path(path: Path) -> Path:
    """Validates that a path is within the data directory."""
    try:
        data_dir = Path(os.getcwd()) / 'data'
        abs_path = path.resolve()
        if not str(abs_path).startswith(str(data_dir)):
            raise HTTPException(
                status_code=403,
                detail="Access denied: Path must be within data directory"
            )
        return abs_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}") 