from pydantic import BaseModel
from enum import Enum
from typing import Optional

class TaskType(Enum):
    API_FETCH = "api_fetch"
    GIT_OPERATION = "git_operation"
    SQL_QUERY = "sql_query"
    WEB_SCRAPE = "web_scrape"
    IMAGE_PROCESS = "image_process"
    AUDIO_TRANSCRIBE = "audio_transcribe"
    MARKDOWN_CONVERT = "markdown_convert"
    CSV_FILTER = "csv_filter"
    CUSTOM = "custom"

class TaskRequest(BaseModel):
    task: str
    task_type: Optional[TaskType] = TaskType.CUSTOM
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None
    input_path: Optional[str] = None  # Must be within /data
    output_path: Optional[str] = None  # Must be within /data


class RunTaskRequest(BaseModel):
    task: str

# No need for RunTaskResponse as we're returning dict directly 