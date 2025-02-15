import io
from PIL import Image
from core.config import get_data_dir
from models.task import RunTaskRequest, RunTaskResponse
import os
from pathlib import Path
import subprocess
import json
from typing import Dict, Any
import logging
from fastapi import HTTPException
from datetime import datetime
import re


async def execute_task(request: RunTaskRequest) -> dict:
    """Executes a task based on the request."""
    try:
        task = request.task.lower()  # Convert to lowercase for easier matching
        data_dir = Path(os.getcwd()) / 'data'
        
        # Handle markdown formatting task (A2)
        if any(keyword in task for keyword in ["format", "prettier"]) and ".md" in task:
            # Extract file path using regex
            file_match = re.search(r'/data/([^\s]+\.md)', task)
            if not file_match:
                raise HTTPException(status_code=400, detail="No markdown file specified")
                
            file_path = data_dir / file_match.group(1)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                
            try:
                result = subprocess.run(
                    ["npx", "prettier@3.4.2", "--write", str(file_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return {"status": "success"}  # Simple success response
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=str(e.stderr))
                
        # Handle Wednesday counting task (A3)
        elif all(keyword in task for keyword in ["dates.txt", "wednesday"]):
            dates_file = data_dir / "dates.txt"
            output_file = data_dir / "dates-wednesdays.txt"
            
            if not dates_file.exists():
                raise HTTPException(status_code=404, detail="dates.txt not found")
                
            with open(dates_file) as f:
                dates = f.readlines()
            
            wednesday_count = sum(
                1 for date in dates 
                if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2
            )
            
            with open(output_file, 'w') as f:
                f.write(str(wednesday_count))
                
            return {"status": "success"}
            
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Task not recognized: {task}"
            )
            
    except Exception as e:
        logging.error(f"Task execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 