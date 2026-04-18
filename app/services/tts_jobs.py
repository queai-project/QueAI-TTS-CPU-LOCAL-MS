from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import pathlib
import queue
import shutil
import threading
import uuid

from fastapi import HTTPException

from app.core.logger import logger
from app.services.tts_service import TTSService


@dataclass
class TTSJob:
    job_id: str
    kind: str
    voice: str
    input_path: pathlib.Path
    output_path: pathlib.Path
    status: str
    created_at: datetime
    voice_settings: dict | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    preview: str | None = None
    original_filename: str | None = None


class TTSJobManager:
    def __init__(self, jobs_dir: str | pathlib.Path = "runtime/tts_jobs"):
        self.jobs_dir = pathlib.Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, TTSJob] = {}
        self._jobs_lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def start(self):
        if self._worker and self._worker.is_alive():
            return

        self._cleanup_all_storage()
        self._worker = threading.Thread(
            target=self._run_worker,
            name="tts-job-worker",
            daemon=True,
        )
        self._worker.start()
        logger.info("TTS job worker started")

    def stop(self):
        if not self._worker:
            return

        self._queue.put(None)
        self._worker.join(timeout=5)
        self._worker = None
        logger.info("TTS job worker stopped")

    def submit_text(
        self,
        text: str,
        voice: str,
        voice_settings: dict | None = None,
    ) -> TTSJob:
        service = TTSService(voice)
        service.validate_voice()
        clean_text = service.validate_text(text)
        normalized_settings = TTSService.normalize_voice_settings(voice_settings)

        job = self._create_job(
            kind="text",
            voice=voice,
            input_suffix=".txt",
            voice_settings=normalized_settings,
            preview=clean_text,
            original_filename=None,
        )
        job.input_path.write_text(clean_text, encoding="utf-8")
        self._queue.put(job.job_id)
        return job

    def submit_document(
        self,
        content: bytes,
        filename: str | None,
        content_type: str | None,
        voice: str,
        voice_settings: dict | None = None,
    ) -> TTSJob:
        service = TTSService(voice)
        service.validate_voice()

        if not content:
            raise HTTPException(status_code=422, detail="The uploaded file is empty")

        extension = service.resolve_document_extension(filename or "document", content_type)
        normalized_settings = TTSService.normalize_voice_settings(voice_settings)

        preview = filename or f"document{extension}"
        job = self._create_job(
            kind="document",
            voice=voice,
            input_suffix=extension,
            voice_settings=normalized_settings,
            preview=preview,
            original_filename=filename,
        )
        job.input_path.write_bytes(content)
        self._queue.put(job.job_id)
        return job

    def get_job(self, job_id: str) -> TTSJob:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    def get_status(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "kind": job.kind,
            "voice": job.voice,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "preview": job.preview,
            "original_filename": job.original_filename,
            "voice_settings": job.voice_settings,
            "result_ready": job.status == "done" and job.output_path.exists(),
        }

    def get_result_path(self, job_id: str) -> pathlib.Path:
        job = self.get_job(job_id)
        if job.status == "failed":
            raise HTTPException(status_code=409, detail=job.error or "Job failed")
        if job.status != "done" or not job.output_path.exists():
            raise HTTPException(status_code=409, detail="Job result is not ready yet")
        return job.output_path

    def cleanup_finished_jobs(self) -> dict:
        with self._jobs_lock:
            removable_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in {"done", "failed"}
            ]

        for job_id in removable_ids:
            self._delete_job(job_id)

        return {"removed_jobs": len(removable_ids)}

    def _create_job(
        self,
        kind: str,
        voice: str,
        input_suffix: str,
        voice_settings: dict | None,
        preview: str | None,
        original_filename: str | None,
    ) -> TTSJob:
        job_id = uuid.uuid4().hex
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = TTSJob(
            job_id=job_id,
            kind=kind,
            voice=voice,
            input_path=job_dir / f"input{input_suffix}",
            output_path=job_dir / "output.wav",
            status="queued",
            created_at=datetime.now(timezone.utc),
            voice_settings=voice_settings,
            preview=preview,
            original_filename=original_filename,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job
        return job

    def _run_worker(self):
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str):
        job = self.get_job(job_id)
        self._update_job(
            job_id,
            status="processing",
            started_at=datetime.now(timezone.utc),
            error=None,
        )

        try:
            service = TTSService(job.voice)
            service.generate_speech_from_input_path(
                job.input_path,
                job.output_path,
                voice_settings=job.voice_settings,
            )
            self._update_job(
                job_id,
                status="done",
                finished_at=datetime.now(timezone.utc),
            )
            logger.info("TTS job %s finished", job_id)
        except Exception as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=datetime.now(timezone.utc),
                error=str(exc),
            )
            logger.exception("TTS job %s failed", job_id)

    def _update_job(self, job_id: str, **changes):
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for field_name, value in changes.items():
                setattr(job, field_name, value)

    def _delete_job(self, job_id: str):
        with self._jobs_lock:
            job = self._jobs.pop(job_id, None)

        if not job:
            return

        job_dir = job.input_path.parent
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

    def _cleanup_all_storage(self):
        removed = 0
        for item in self.jobs_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                removed += 1
            elif item.is_file():
                item.unlink(missing_ok=True)
                removed += 1

        if removed:
            logger.info("Removed %s stale TTS job directories on startup", removed)