#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

from huggingface_hub import snapshot_download


REPO_ID = "rhasspy/piper-voices"


@dataclass(frozen=True)
class VoiceInfo:
    full_name: str
    lang: str
    dialect: str
    speaker: str
    quality: str

    @property
    def relative_dir(self) -> pathlib.Path:
        return pathlib.Path(self.lang) / self.dialect / self.speaker / self.quality

    @property
    def onnx_filename(self) -> str:
        return f"{self.full_name}.onnx"

    @property
    def json_filename(self) -> str:
        return f"{self.full_name}.onnx.json"

    @property
    def onnx_relative_path(self) -> pathlib.Path:
        return self.relative_dir / self.onnx_filename

    @property
    def json_relative_path(self) -> pathlib.Path:
        return self.relative_dir / self.json_filename


class PiperVoiceDownloader:
    def __init__(self, voices_dir: pathlib.Path, voice_script: pathlib.Path) -> None:
        self.voices_dir = voices_dir.resolve()
        self.voice_script = voice_script.resolve()
        self.base_dir = self.voices_dir.parent.resolve()

    def get_listed_voices(self) -> list[str]:
        if not self.voice_script.exists():
            raise FileNotFoundError(f"No existe el script: {self.voice_script}")

        output = subprocess.check_output(
            [str(self.voice_script)],
            text=True,
            stderr=subprocess.STDOUT,
            cwd=str(self.base_dir),
        )

        voices = sorted({line.strip() for line in output.splitlines() if line.strip()})
        return voices

    def parse_voice(self, voice_name: str) -> VoiceInfo:
        try:
            dialect, speaker, quality = voice_name.split("-", 2)
            lang = dialect.split("_", 1)[0]
        except ValueError as exc:
            raise ValueError(
                f"Formato de voz inválido: '{voice_name}'. "
                "Se esperaba algo como 'en_US-amy-medium'."
            ) from exc

        return VoiceInfo(
            full_name=voice_name,
            lang=lang,
            dialect=dialect,
            speaker=speaker,
            quality=quality,
        )

    def ensure_parent_dir(self, voice: VoiceInfo) -> None:
        (self.voices_dir / voice.relative_dir).mkdir(parents=True, exist_ok=True)

    def remove_voice_dir(self, voice: VoiceInfo) -> None:
        target_dir = self.voices_dir / voice.relative_dir
        if target_dir.exists():
            shutil.rmtree(target_dir)

    def remove_broken_placeholders(self, voice: VoiceInfo) -> None:
        """
        Borra archivos claramente inválidos descargados de forma incorrecta.
        Un .onnx real suele pesar muchos MB; un placeholder/LFS roto suele ser muy pequeño.
        """
        onnx_path = self.voices_dir / voice.onnx_relative_path
        json_path = self.voices_dir / voice.json_relative_path

        if onnx_path.exists() and onnx_path.stat().st_size < 1024 * 1024:
            onnx_path.unlink()

        if json_path.exists() and json_path.stat().st_size < 100:
            json_path.unlink()

    def download_voice_folder(self, voice: VoiceInfo, force: bool = False) -> None:
        if force:
            self.remove_voice_dir(voice)
        else:
            self.remove_broken_placeholders(voice)

        self.ensure_parent_dir(voice)

        local_cache_dir = self.voices_dir / ".cache" / "huggingface"
        local_cache_dir.mkdir(parents=True, exist_ok=True)

        allow_patterns = [f"{voice.relative_dir.as_posix()}/**"]

        import os

        # Evita problemas con Xet/CAS y evita escribir en ~/.cache
        os.environ["HF_HOME"] = str(local_cache_dir)
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        # Un solo intento "normal"
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                local_dir=str(self.voices_dir),
                cache_dir=str(local_cache_dir),
                allow_patterns=allow_patterns,
                max_workers=2,
            )
        except Exception:
            # Reintento más conservador
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                local_dir=str(self.voices_dir),
                cache_dir=str(local_cache_dir),
                allow_patterns=allow_patterns,
                max_workers=1,
            )

        self.validate_download(voice)

    def validate_download(self, voice: VoiceInfo) -> None:
        onnx_path = self.voices_dir / voice.onnx_relative_path
        json_path = self.voices_dir / voice.json_relative_path

        if not onnx_path.exists():
            raise FileNotFoundError(f"No se descargó el modelo: {onnx_path}")

        if not json_path.exists():
            raise FileNotFoundError(f"No se descargó la config: {json_path}")

        if onnx_path.stat().st_size < 1024 * 1024:
            raise RuntimeError(
                f"El archivo parece inválido o incompleto: {onnx_path} "
                f"({onnx_path.stat().st_size} bytes)"
            )

        if json_path.stat().st_size < 50:
            raise RuntimeError(
                f"La config parece inválida o incompleta: {json_path} "
                f"({json_path.stat().st_size} bytes)"
            )
        
        
    def download_listed_voices(
        self,
        only: list[str] | None = None,
        force: bool = False,
    ) -> None:
        listed = self.get_listed_voices()

        if only:
            listed_set = set(listed)
            missing = [voice for voice in only if voice not in listed_set]
            if missing:
                raise ValueError(
                    "Estas voces no aparecen en tu listado local: "
                    + ", ".join(sorted(missing))
                )
            target_voices = only
        else:
            target_voices = listed

        if not target_voices:
            print("No hay voces listadas para descargar.")
            return

        print(f"Directorio de voces: {self.voices_dir}")
        print(f"Total de voces a descargar: {len(target_voices)}")

        for index, voice_name in enumerate(target_voices, start=1):
            voice = self.parse_voice(voice_name)
            print(f"[{index}/{len(target_voices)}] Descargando {voice_name} ...")
            self.download_voice_folder(voice, force=force)
            print(f"  OK -> {voice.relative_dir}")


def build_default_paths() -> tuple[pathlib.Path, pathlib.Path]:
    script_path = pathlib.Path(__file__).resolve()
    voices_dir = script_path.parents[1]
    voice_script = voices_dir / "_script" / "voice_names.sh"
    return voices_dir, voice_script


def main() -> int:
    default_voices_dir, default_voice_script = build_default_paths()

    parser = argparse.ArgumentParser(
        description=(
            "Descarga desde Hugging Face las carpetas completas de las voces "
            "listadas por voice_names.sh."
        )
    )
    parser.add_argument(
        "--voices-dir",
        default=str(default_voices_dir),
        help="Directorio raíz de voices",
    )
    parser.add_argument(
        "--voice-script",
        default=str(default_voice_script),
        help="Ruta al script voice_names.sh",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Descargar solo estas voces (ej: en_US-amy-medium es_MX-claude-high)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borra la carpeta local de cada voz antes de descargarla otra vez",
    )

    args = parser.parse_args()

    downloader = PiperVoiceDownloader(
        voices_dir=pathlib.Path(args.voices_dir),
        voice_script=pathlib.Path(args.voice_script),
    )

    try:
        downloader.download_listed_voices(
            only=args.only,
            force=args.force,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Descarga completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())