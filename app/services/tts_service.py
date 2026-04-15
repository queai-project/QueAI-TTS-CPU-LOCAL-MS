from io import BytesIO
import pathlib
import subprocess

from fastapi import HTTPException


class TTSService:
    SUPPORTED_DOCUMENT_TYPES = {
        ".txt": "plain text",
        ".md": "Markdown",
        ".pdf": "PDF",
    }

    BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
    VOICES_DIR = BASE_DIR / "voices"
    VOICE_SCRIPT = VOICES_DIR / "_script" / "voice_names.sh"

    def __init__(self, voice=None):
        self.voice = voice
        self.available_voices: dict = self._get_available_voices()

    def validate_text(self, text: str) -> str:
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="Text empty")
        return text.strip()

    def validate_voice(self):
        if not self.voice or not self.voice.strip():
            raise HTTPException(status_code=422, detail="Voice empty")
        if self.voice not in self.available_voices["voices"]:
            raise HTTPException(status_code=404, detail="Voice not found")

    def generate_speech_from_input_path(
        self,
        input_path: str | pathlib.Path,
        output_path: str | pathlib.Path,
    ) -> pathlib.Path:
        self.validate_voice()
        text = self._extract_text_from_path(pathlib.Path(input_path))
        text = self.validate_text(text)
        return self._generate_to_path(text, pathlib.Path(output_path))

    def _get_available_voices(self) -> dict:
        try:
            result = subprocess.check_output(
                [str(self.VOICE_SCRIPT)],
                text=True,
                stderr=subprocess.STDOUT,
                cwd=str(self.BASE_DIR),
            )
            voice_list = [line.strip() for line in result.split("\n") if line.strip()]
            return {"voices": voice_list}
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Script failed: {e.output}")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Script not found: {self.VOICE_SCRIPT}",
            )

    def _generate_to_path(self, text: str, output_path: pathlib.Path) -> pathlib.Path:
        model_path, config_path = self._get_model_and_config()
        import wave
        from piper.voice import PiperVoice

        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = PiperVoice.load(model_path, config_path)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            voice.synthesize_wav(text, wav_file)

        return output_path

    def _get_path_voice(self) -> pathlib.Path:
        dialect, name, quality = self.voice.split("-")
        lang, _ = dialect.split("_")
        return self.VOICES_DIR / lang / dialect / name / quality / self.voice

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