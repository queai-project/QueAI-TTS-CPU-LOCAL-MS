from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import sys
import time
from io import BytesIO
from typing import Any

from fastapi import HTTPException
from huggingface_hub import HfApi


class TTSService:
    SUPPORTED_DOCUMENT_TYPES = {
        ".txt": "plain text",
        ".md": "Markdown",
        ".pdf": "PDF",
    }

    REPO_ID = "rhasspy/piper-voices"
    HF_RESOLVE_BASE = f"https://huggingface.co/{REPO_ID}/resolve/main"

    BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
    VOICES_DIR = BASE_DIR / "voices"
    DOWNLOAD_SCRIPT = VOICES_DIR / "_script" / "download_listed_voices.py"

    CACHE_DIR = VOICES_DIR / ".cache"
    CATALOG_CACHE_PATH = CACHE_DIR / "voice_catalog.json"

    _catalog_lock: asyncio.Lock | None = None
    _download_lock: asyncio.Lock | None = None
    _download_jobs: dict[str, dict[str, Any]] = {}
    _download_tasks: dict[str, asyncio.Task] = {}

    LANGUAGE_LABELS = {
        "ar": "Árabe",
        "bg": "Búlgaro",
        "ca": "Catalán",
        "cs": "Checo",
        "cy": "Galés",
        "da": "Danés",
        "de": "Alemán",
        "el": "Griego",
        "en": "Inglés",
        "es": "Español",
        "eu": "Euskera",
        "fa": "Persa",
        "fi": "Finés",
        "fr": "Francés",
        "hi": "Hindi",
        "hu": "Húngaro",
        "id": "Indonesio",
        "is": "Islandés",
        "it": "Italiano",
        "ka": "Georgiano",
        "kk": "Kazajo",
        "ku": "Kurdo",
        "lb": "Luxemburgués",
        "lv": "Letón",
        "ml": "Malayalam",
        "ne": "Nepalí",
        "nl": "Neerlandés",
        "no": "Noruego",
        "pl": "Polaco",
        "pt": "Portugués",
        "ro": "Rumano",
        "ru": "Ruso",
        "sk": "Eslovaco",
        "sl": "Esloveno",
        "sq": "Albanés",
        "sr": "Serbio",
        "sv": "Sueco",
        "sw": "Suajili",
        "te": "Telugu",
        "tr": "Turco",
        "uk": "Ucraniano",
        "ur": "Urdu",
        "vi": "Vietnamita",
        "zh": "Chino",
    }

    DEFAULT_INFERENCE_LIMITS = {
        "noise_scale": {"min": 0.0, "max": 2.0, "step": 0.01},
        "length_scale": {"min": 0.1, "max": 3.0, "step": 0.01},
        "noise_w": {"min": 0.0, "max": 2.0, "step": 0.01},
    }

    def __init__(self, voice: str | None = None):
        self.voice = voice
        self.available_voices: dict = self._get_available_voices()

    @classmethod
    def _get_catalog_lock(cls) -> asyncio.Lock:
        if cls._catalog_lock is None:
            cls._catalog_lock = asyncio.Lock()
        return cls._catalog_lock

    @classmethod
    def _get_download_lock(cls) -> asyncio.Lock:
        if cls._download_lock is None:
            cls._download_lock = asyncio.Lock()
        return cls._download_lock

    @classmethod
    def _parse_voice_name(cls, voice_name: str) -> dict[str, str]:
        try:
            dialect, speaker, quality = voice_name.split("-", 2)
            lang = dialect.split("_", 1)[0]
            return {
                "voice": voice_name,
                "lang": lang,
                "dialect": dialect,
                "speaker": speaker,
                "quality": quality,
            }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid voice format: {voice_name}") from exc

    @classmethod
    def _get_voice_base_path(cls, voice_name: str) -> pathlib.Path:
        parts = cls._parse_voice_name(voice_name)
        return cls.VOICES_DIR / parts["lang"] / parts["dialect"] / parts["speaker"] / parts["quality"] / parts["voice"]

    @classmethod
    def _get_voice_dir(cls, voice_name: str) -> pathlib.Path:
        parts = cls._parse_voice_name(voice_name)
        return cls.VOICES_DIR / parts["lang"] / parts["dialect"] / parts["speaker"] / parts["quality"]

    @classmethod
    def _voice_model_exists_and_valid(cls, voice_name: str) -> bool:
        base = cls._get_voice_base_path(voice_name)
        model_path = pathlib.Path(str(base) + ".onnx")
        config_path = pathlib.Path(str(base) + ".onnx.json")

        if not model_path.exists() or not config_path.exists():
            return False

        try:
            if model_path.stat().st_size < 1024 * 1024:
                return False
            if config_path.stat().st_size < 50:
                return False
        except OSError:
            return False

        return True

    @classmethod
    def _voice_lite_exists_and_valid(cls, voice_name: str) -> bool:
        base = cls._get_voice_base_path(voice_name)
        config_path = pathlib.Path(str(base) + ".onnx.json")
        if not config_path.exists():
            return False
        try:
            return config_path.stat().st_size >= 50
        except OSError:
            return False

    @classmethod
    def _get_local_sample_path(cls, voice_name: str) -> pathlib.Path | None:
        voice_dir = cls._get_voice_dir(voice_name)
        samples_dir = voice_dir / "samples"
        if not samples_dir.exists():
            return None

        preferred = samples_dir / "speaker_0.mp3"
        if preferred.exists():
            return preferred

        samples = sorted(samples_dir.glob("*.mp3"))
        return samples[0] if samples else None

    def validate_text(self, text: str) -> str:
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="Text empty")
        return text.strip()

    def validate_voice(self):
        if not self.voice or not self.voice.strip():
            raise HTTPException(status_code=422, detail="Voice empty")

        if not self._voice_model_exists_and_valid(self.voice):
            raise HTTPException(
                status_code=409,
                detail="Voice model is not downloaded locally. Download the voice first.",
            )

    @classmethod
    def normalize_voice_settings(cls, voice_settings: dict[str, Any] | None) -> dict[str, float]:
        normalized = {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
        }

        if not voice_settings:
            return normalized

        for key, limits in cls.DEFAULT_INFERENCE_LIMITS.items():
            if key not in voice_settings or voice_settings[key] is None:
                continue
            try:
                value = float(voice_settings[key])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid value for {key}") from exc

            if value < limits["min"] or value > limits["max"]:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{key} out of range. "
                        f"Allowed range: {limits['min']} to {limits['max']}"
                    ),
                )
            normalized[key] = value

        return normalized

    def generate_speech_from_input_path(
        self,
        input_path: str | pathlib.Path,
        output_path: str | pathlib.Path,
        voice_settings: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        self.validate_voice()
        text = self._extract_text_from_path(pathlib.Path(input_path))
        text = self.validate_text(text)
        return self._generate_to_path(text, pathlib.Path(output_path), voice_settings=voice_settings)

    def _generate_to_path(
        self,
        text: str,
        output_path: pathlib.Path,
        voice_settings: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        model_path, config_path = self._get_model_and_config()
        import wave
        from piper.voice import PiperVoice

        settings = self.normalize_voice_settings(voice_settings)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = PiperVoice.load(model_path, config_path)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)

            synthesis_config = None
            with contextlib.suppress(Exception):
                from piper.voice import SynthesisConfig

                synthesis_config = SynthesisConfig()
                synthesis_config.noise_scale = settings["noise_scale"]
                synthesis_config.length_scale = settings["length_scale"]

                if hasattr(synthesis_config, "noise_w"):
                    synthesis_config.noise_w = settings["noise_w"]
                elif hasattr(synthesis_config, "noise_w_scale"):
                    synthesis_config.noise_w_scale = settings["noise_w"]

            if synthesis_config is not None:
                voice.synthesize_wav(text, wav_file, synthesis_config)
            else:
                voice.synthesize_wav(text, wav_file)

        return output_path

    def _get_path_voice(self) -> pathlib.Path:
        parts = self._parse_voice_name(self.voice)
        return self.VOICES_DIR / parts["lang"] / parts["dialect"] / parts["speaker"] / parts["quality"] / parts["voice"]

    def _get_model_and_config(self):
        path = self._get_path_voice()
        model_path = str(path) + ".onnx"
        config_path = str(path) + ".onnx.json"
        return model_path, config_path

    def resolve_document_extension(self, filename: str, content_type: str | None) -> str:
        extension = pathlib.Path(filename).suffix.lower()
        if extension in self.SUPPORTED_DOCUMENT_TYPES:
            return extension

        mime_to_extension = {
            "text/plain": ".txt",
            "text/markdown": ".md",
            "text/x-markdown": ".md",
            "application/pdf": ".pdf",
        }
        guessed_extension = mime_to_extension.get((content_type or "").lower(), "")
        if guessed_extension:
            return guessed_extension

        supported_types = ", ".join(sorted(self.SUPPORTED_DOCUMENT_TYPES))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported document type. Supported types: {supported_types}",
        )

    def _extract_text_from_path(self, input_path: pathlib.Path) -> str:
        raw_content = input_path.read_bytes()
        extension = input_path.suffix.lower()

        if extension in {".txt", ".md"}:
            return self._extract_text_from_text_like(raw_content)
        if extension == ".pdf":
            return self._extract_text_from_pdf(raw_content)

        supported_types = ", ".join(sorted(self.SUPPORTED_DOCUMENT_TYPES))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported document type. Supported types: {supported_types}",
        )

    def _extract_text_from_text_like(self, raw_content: bytes) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=422, detail="Could not decode the text file")

    def _extract_text_from_pdf(self, raw_content: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="PDF support is not installed on the server",
            ) from exc

        try:
            reader = PdfReader(BytesIO(raw_content))
            return "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Could not read the PDF file") from exc

    @classmethod
    def _load_catalog_cache(cls) -> dict[str, Any] | None:
        if not cls.CATALOG_CACHE_PATH.exists():
            return None
        try:
            return json.loads(cls.CATALOG_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    @classmethod
    def _save_catalog_cache(cls, data: dict[str, Any]) -> None:
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CATALOG_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _ensure_language_dirs_from_catalog(cls, voices_map: dict[str, dict[str, Any]]) -> None:
        langs = sorted({entry["lang"] for entry in voices_map.values()})
        for lang in langs:
            (cls.VOICES_DIR / lang).mkdir(parents=True, exist_ok=True)

    @classmethod
    def _build_remote_catalog_sync(cls) -> dict[str, Any]:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=cls.REPO_ID, repo_type="model")
        voices_map: dict[str, dict[str, Any]] = {}

        for repo_path in repo_files:
            if not repo_path.endswith(".onnx") or repo_path.endswith(".onnx.json"):
                continue

            parts = pathlib.PurePosixPath(repo_path).parts
            if len(parts) < 5:
                continue

            lang, dialect, speaker, quality, filename = parts[:5]
            full_name = filename[:-5]
            expected = f"{dialect}-{speaker}-{quality}"

            if full_name != expected:
                continue

            voices_map[full_name] = {
                "voice": full_name,
                "lang": lang,
                "dialect": dialect,
                "speaker": speaker,
                "quality": quality,
                "repo_dir": f"{lang}/{dialect}/{speaker}/{quality}",
                "onnx_repo_path": repo_path,
                "json_repo_path": f"{lang}/{dialect}/{speaker}/{quality}/{full_name}.onnx.json",
                "sample_repo_path": None,
            }

        for repo_path in repo_files:
            if "/samples/" not in repo_path or not repo_path.lower().endswith(".mp3"):
                continue

            parts = pathlib.PurePosixPath(repo_path).parts
            if len(parts) < 6:
                continue

            lang, dialect, speaker, quality = parts[0], parts[1], parts[2], parts[3]
            voice_name = f"{dialect}-{speaker}-{quality}"

            if voice_name not in voices_map:
                continue

            current = voices_map[voice_name]["sample_repo_path"]
            if current is None or repo_path.endswith("/speaker_0.mp3"):
                voices_map[voice_name]["sample_repo_path"] = repo_path

        cls._ensure_language_dirs_from_catalog(voices_map)

        voices: list[dict[str, Any]] = []
        languages_map: dict[str, dict[str, Any]] = {}

        for voice_name in sorted(voices_map.keys(), key=lambda v: (
            voices_map[v]["lang"],
            voices_map[v]["dialect"],
            voices_map[v]["speaker"],
            voices_map[v]["quality"],
        )):
            entry = voices_map[voice_name]
            lite_available = cls._voice_lite_exists_and_valid(voice_name)
            model_available = cls._voice_model_exists_and_valid(voice_name)

            voice_payload = {
                "voice": voice_name,
                "lang": entry["lang"],
                "lang_label": cls.LANGUAGE_LABELS.get(entry["lang"], entry["lang"].upper()),
                "dialect": entry["dialect"],
                "speaker": entry["speaker"],
                "quality": entry["quality"],
                "lite_available": lite_available,
                "model_available": model_available,
                "sample_available": bool(entry.get("sample_repo_path")),
            }
            voices.append(voice_payload)

            bucket = languages_map.setdefault(
                entry["lang"],
                {
                    "code": entry["lang"],
                    "label": cls.LANGUAGE_LABELS.get(entry["lang"], entry["lang"].upper()),
                    "voices_count": 0,
                    "lite_count": 0,
                    "model_count": 0,
                    "folder_exists": (cls.VOICES_DIR / entry["lang"]).exists(),
                },
            )
            bucket["voices_count"] += 1
            if lite_available:
                bucket["lite_count"] += 1
            if model_available:
                bucket["model_count"] += 1

        return {
            "languages": sorted(languages_map.values(), key=lambda item: item["label"]),
            "voices": voices,
            "generated_at": time.time(),
        }

    @classmethod
    async def get_voice_catalog(cls, refresh: bool = False) -> dict[str, Any]:
        async with cls._get_catalog_lock():
            if not refresh:
                cached = cls._load_catalog_cache()
                if cached:
                    for voice in cached.get("voices", []):
                        voice["lite_available"] = cls._voice_lite_exists_and_valid(voice["voice"])
                        voice["model_available"] = cls._voice_model_exists_and_valid(voice["voice"])

                    languages_map: dict[str, dict[str, Any]] = {}
                    for voice in cached.get("voices", []):
                        bucket = languages_map.setdefault(
                            voice["lang"],
                            {
                                "code": voice["lang"],
                                "label": cls.LANGUAGE_LABELS.get(voice["lang"], voice["lang"].upper()),
                                "voices_count": 0,
                                "lite_count": 0,
                                "model_count": 0,
                                "folder_exists": (cls.VOICES_DIR / voice["lang"]).exists(),
                            },
                        )
                        bucket["voices_count"] += 1
                        if voice["lite_available"]:
                            bucket["lite_count"] += 1
                        if voice["model_available"]:
                            bucket["model_count"] += 1

                    cached["languages"] = sorted(languages_map.values(), key=lambda item: item["label"])
                    return cached

            catalog = await asyncio.to_thread(cls._build_remote_catalog_sync)
            cls._save_catalog_cache(catalog)
            return catalog

    @classmethod
    async def get_voice_catalog_with_runtime(cls, refresh: bool = False) -> dict[str, Any]:
        catalog = await cls.get_voice_catalog(refresh=refresh)
        downloads = {}

        for voice in catalog.get("voices", []):
            voice_key = f"voice:{voice['voice']}"
            downloads[voice_key] = cls.get_download_status_sync(voice_key)

        for lang in catalog.get("languages", []):
            lang_key = f"lang:{lang['code']}"
            downloads[lang_key] = cls.get_download_status_sync(lang_key)

        downloads["__bootstrap__"] = cls.get_download_status_sync("__bootstrap__")

        return {
            **catalog,
            "downloads": downloads,
        }

    @classmethod
    def _read_voice_json(cls, voice_name: str) -> dict[str, Any]:
        base = cls._get_voice_base_path(voice_name)
        config_path = pathlib.Path(str(base) + ".onnx.json")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Voice JSON not downloaded locally")
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not read voice JSON") from exc

    @classmethod
    def get_configurable_voice_settings(cls, voice_name: str) -> dict[str, Any]:
        data = cls._read_voice_json(voice_name)
        inference = data.get("inference", {}) or {}

        defaults = {
            "noise_scale": float(inference.get("noise_scale", 0.667)),
            "length_scale": float(inference.get("length_scale", 1.0)),
            "noise_w": float(inference.get("noise_w", 0.8)),
        }

        language = data.get("language", {}) or {}
        audio = data.get("audio", {}) or {}

        return {
            "voice": voice_name,
            "voice_name": cls._parse_voice_name(voice_name)["speaker"],
            "language": {
                "code": language.get("code"),
                "family": language.get("family"),
                "region": language.get("region"),
                "name_native": language.get("name_native"),
                "name_english": language.get("name_english"),
                "country_english": language.get("country_english"),
            },
            "audio": {
                "sample_rate": audio.get("sample_rate"),
                "quality": audio.get("quality"),
            },
            "configurable": {
                "noise_scale": {
                    "label": "Naturalidad",
                    "description": "Hace que la voz suene más estable o más variada",
                    "value": defaults["noise_scale"],
                    **cls.DEFAULT_INFERENCE_LIMITS["noise_scale"],
                },
                "length_scale": {
                    "label": "Velocidad",
                    "description": "Más bajo = más rápido; más alto = más lento",
                    "value": defaults["length_scale"],
                    **cls.DEFAULT_INFERENCE_LIMITS["length_scale"],
                },
                "noise_w": {
                    "label": "Claridad",
                    "description": "Ajusta cómo de marcada y cambiante suena la pronunciación",
                    "value": defaults["noise_w"],
                    **cls.DEFAULT_INFERENCE_LIMITS["noise_w"],
                },
            },
        }

    def _get_available_voices(self) -> dict:
        catalog = self._load_catalog_cache() or {"voices": []}
        local_models = [
            voice["voice"]
            for voice in catalog.get("voices", [])
            if self._voice_model_exists_and_valid(voice["voice"])
        ]
        return {"voices": sorted(local_models)}

    @classmethod
    async def get_voice_sample_source(cls, voice_name: str) -> dict[str, str] | None:
        catalog = await cls.get_voice_catalog(refresh=False)
        entry = next((item for item in catalog.get("voices", []) if item["voice"] == voice_name), None)
        if not entry:
            return None

        local_sample = cls._get_local_sample_path(voice_name)
        if local_sample and local_sample.exists():
            return {"type": "local", "value": str(local_sample)}

        parts = cls._parse_voice_name(voice_name)
        remote_sample = (
            f"{cls.HF_RESOLVE_BASE}/"
            f"{parts['lang']}/{parts['dialect']}/{parts['speaker']}/{parts['quality']}/samples/speaker_0.mp3"
        )
        return {"type": "remote", "value": remote_sample}

    @classmethod
    def get_download_status_sync(cls, key: str) -> dict[str, Any]:
        existing = cls._download_jobs.get(key)
        if existing:
            return existing

        return {
            "key": key,
            "status": "idle",
            "progress": 0.0,
            "message": "Idle",
            "current_file": None,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "mode": None,
            "kind": None,
        }

    @classmethod
    async def get_download_status(cls, key: str) -> dict[str, Any]:
        return cls.get_download_status_sync(key)

    @classmethod
    async def bootstrap_initial_voices(cls) -> dict[str, Any]:
        key = "__bootstrap__"
        return await cls._start_download(
            key=key,
            command=[
                sys.executable,
                str(cls.DOWNLOAD_SCRIPT),
                "--json-progress",
            ],
            kind="bootstrap",
            mode="mixed",
        )

    @classmethod
    async def download_language_catalog(cls, lang: str) -> dict[str, Any]:
        key = f"lang:{lang}"
        return await cls._start_download(
            key=key,
            command=[
                sys.executable,
                str(cls.DOWNLOAD_SCRIPT),
                "--lang",
                lang,
                "--json-progress",
            ],
            kind="language",
            mode="lite",
        )

    @classmethod
    async def download_voice(cls, voice_name: str, mode: str = "model") -> dict[str, Any]:
        if mode not in {"full", "lite", "model"}:
            raise HTTPException(status_code=422, detail="Invalid download mode")

        key = f"voice:{voice_name}"
        return await cls._start_download(
            key=key,
            command=[
                sys.executable,
                str(cls.DOWNLOAD_SCRIPT),
                "--voice",
                voice_name,
                "--mode",
                mode,
                "--json-progress",
            ],
            kind="voice",
            mode=mode,
        )

    @classmethod
    async def _start_download(cls, key: str, command: list[str], kind: str, mode: str) -> dict[str, Any]:
        async with cls._get_download_lock():
            current = cls._download_jobs.get(key)
            task = cls._download_tasks.get(key)

            if task and not task.done():
                return current

            cls._download_jobs[key] = {
                "key": key,
                "status": "queued",
                "progress": 0.0,
                "message": "Queued",
                "current_file": None,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "mode": mode,
                "kind": kind,
            }

            cls._download_tasks[key] = asyncio.create_task(
                cls._run_download_subprocess(key, command)
            )
            return cls._download_jobs[key]

    @classmethod
    async def _run_download_subprocess(cls, key: str, command: list[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cls.BASE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue

                try:
                    payload = json.loads(text)
                    cls._apply_download_event(key, payload)
                except json.JSONDecodeError:
                    cls._download_jobs[key] = {
                        **cls._download_jobs.get(key, {}),
                        "key": key,
                        "status": "running",
                        "message": text,
                    }

            return_code = await process.wait()

            if return_code == 0:
                cls._download_jobs[key] = {
                    **cls._download_jobs.get(key, {}),
                    "key": key,
                    "status": "done",
                    "progress": 100.0,
                    "message": "Download finished",
                    "current_file": None,
                }
                with contextlib.suppress(Exception):
                    catalog = await cls.get_voice_catalog(refresh=True)
                    cls._save_catalog_cache(catalog)
            else:
                current = cls._download_jobs.get(key, {})
                cls._download_jobs[key] = {
                    **current,
                    "key": key,
                    "status": "failed",
                    "message": current.get("message") or f"Download process failed ({return_code})",
                }

        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            cls._download_jobs[key] = {
                **cls._download_jobs.get(key, {}),
                "key": key,
                "status": "failed",
                "message": "Download cancelled",
            }
            raise
        finally:
            cls._download_tasks.pop(key, None)

    @classmethod
    def _apply_download_event(cls, key: str, payload: dict[str, Any]) -> None:
        event = payload.get("event")
        current = cls._download_jobs.get(key, {})
        merged = {
            **current,
            "key": key,
        }

        if event == "start":
            merged.update(
                {
                    "status": "running",
                    "progress": 0.0,
                    "message": "Starting download",
                    "current_file": None,
                    "downloaded_bytes": 0,
                    "total_bytes": payload.get("total_bytes", 0),
                    "mode": payload.get("mode", current.get("mode")),
                }
            )

        elif event == "file_start":
            merged.update(
                {
                    "status": "running",
                    "message": "Downloading files",
                    "current_file": payload.get("file"),
                }
            )

        elif event == "progress":
            merged.update(
                {
                    "status": "running",
                    "progress": float(payload.get("percent", 0.0)),
                    "message": "Downloading files",
                    "current_file": payload.get("file"),
                    "downloaded_bytes": int(payload.get("overall_bytes", 0)),
                    "total_bytes": int(payload.get("overall_total_bytes", 0)),
                }
            )

        elif event == "file_done":
            merged.update(
                {
                    "status": "running",
                    "message": "Processing next file",
                    "current_file": payload.get("file"),
                }
            )

        elif event in {"done", "bootstrap_done", "language_lite_done"}:
            merged.update(
                {
                    "status": "done",
                    "progress": 100.0,
                    "message": "Download finished",
                    "current_file": None,
                    "downloaded_bytes": int(payload.get("total_bytes", merged.get("downloaded_bytes", 0))),
                    "total_bytes": int(payload.get("total_bytes", merged.get("total_bytes", 0))),
                }
            )

        elif event == "error":
            merged.update(
                {
                    "status": "failed",
                    "message": payload.get("detail", "Download failed"),
                }
            )

        elif event == "cancelled":
            merged.update(
                {
                    "status": "failed",
                    "message": "Download cancelled",
                }
            )

        else:
            merged.update(
                {
                    "status": current.get("status", "running"),
                    "message": payload.get("message", current.get("message", "Running")),
                }
            )

        cls._download_jobs[key] = merged