from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VoiceSettingsRequest(BaseModel):
    noise_scale: float | None = Field(default=None)
    length_scale: float | None = Field(default=None)
    noise_w: float | None = Field(default=None)


class UserTextRequest(BaseModel):
    text: str
    voice: str
    voice_settings: VoiceSettingsRequest | None = None


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    kind: Literal["text", "document"] | str
    voice: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None