from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from app.schemas.tts import (
    JobStatusResponse,
    JobSubmissionResponse,
    UserTextRequest,
    VoiceSettingsRequest,
)
from app.services.tts_service import TTSService


router = APIRouter()


class VoiceDownloadRequest(BaseModel):
    voice: str
    mode: str = "model"


class LanguageDownloadRequest(BaseModel):
    lang: str


@router.post("/text", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmissionResponse)
async def enqueue_speech_from_text(payload: UserTextRequest, request: Request):
    TTSService(payload.voice).validate_voice()
    job = request.app.state.tts_jobs.submit_text(
        payload.text,
        payload.voice,
        voice_settings=payload.voice_settings.model_dump() if payload.voice_settings else None,
    )
    return {"job_id": job.job_id, "status": job.status}


@router.post("/document", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmissionResponse)
async def enqueue_speech_from_document(
    request: Request,
    file: UploadFile = File(...),
    voice: str = Form(...),
    voice_settings_json: str | None = Form(default=None),
):
    TTSService(voice).validate_voice()

    voice_settings = None
    if voice_settings_json:
        try:
            parsed = json.loads(voice_settings_json)
            voice_settings = VoiceSettingsRequest(**parsed).model_dump()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Invalid voice settings JSON") from exc

    try:
        content = await file.read()
    finally:
        await file.close()

    job = request.app.state.tts_jobs.submit_document(
        content,
        file.filename,
        file.content_type,
        voice,
        voice_settings=voice_settings,
    )
    return {"job_id": job.job_id, "status": job.status}


@router.delete("/jobs")
async def cleanup_finished_jobs(request: Request):
    return request.app.state.tts_jobs.cleanup_finished_jobs()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_tts_job(job_id: str, request: Request):
    return request.app.state.tts_jobs.get_status(job_id)


@router.get("/jobs/{job_id}/result")
async def get_tts_job_result(job_id: str, request: Request):
    result_path = request.app.state.tts_jobs.get_result_path(job_id)
    return FileResponse(path=result_path, media_type="audio/wav", filename=f"{job_id}.wav")


@router.get("/voices")
async def get_voices(refresh: bool = False):
    return await TTSService.get_voice_catalog_with_runtime(refresh=refresh)


@router.get("/voices/{voice_name}/sample")
async def get_voice_sample(voice_name: str):
    source = await TTSService.get_voice_sample_source(voice_name)
    if not source:
        raise HTTPException(status_code=404, detail="Sample not found")

    if source["type"] == "local":
        return FileResponse(path=source["value"], media_type="audio/mpeg")

    return RedirectResponse(url=source["value"])


@router.get("/voices/{voice_name}/settings")
async def get_voice_settings(voice_name: str):
    return TTSService.get_configurable_voice_settings(voice_name)


@router.post("/voices/bootstrap")
async def bootstrap_initial_voices():
    return await TTSService.bootstrap_initial_voices()


@router.post("/voices/language")
async def download_language_catalog(payload: LanguageDownloadRequest):
    return await TTSService.download_language_catalog(payload.lang)


@router.post("/voices/download")
async def download_voice(payload: VoiceDownloadRequest):
    return await TTSService.download_voice(payload.voice, payload.mode)


@router.get("/voices/downloads/{key:path}")
async def get_voice_download_status(key: str):
    return await TTSService.get_download_status(key)