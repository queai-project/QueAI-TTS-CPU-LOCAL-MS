from datetime import datetime

from pydantic import BaseModel


class UserTextRequest(BaseModel):
    text: str
    voice: str


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    kind: str
    voice: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result_ready: bool
