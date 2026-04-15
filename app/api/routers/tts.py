from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.schemas.tts import JobStatusResponse, JobSubmissionResponse, UserTextRequest
from app.services.tts_service import TTSService


router = APIRouter()


@router.post("/text", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmissionResponse)
async def enqueue_speech_from_text(payload: UserTextRequest, request: Request):
    job = request.app.state.tts_jobs.submit_text(payload.text, payload.voice)
    return {"job_id": job.job_id, "status": job.status}


@router.post("/document", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmissionResponse)
async def enqueue_speech_from_document(
    request: Request,
    file: UploadFile = File(...),
    voice: str = Form(...),
):
    try:
        content = await file.read()
    finally:
        await file.close()

    job = request.app.state.tts_jobs.submit_document(content, file.filename, file.content_type, voice)
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
async def get_voices():
    return TTSService().available_voices or {}
