import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import faster_whisper

from thesis_demo.config import select_backend

MODEL_NAME = "small"  # faster-whisper model size
FFMPEG_TIMEOUT_S = 30


def default_work_dir():
    """Return the fallback recording directory when no caller supplies one."""
    return Path(tempfile.gettempdir()) / "thesis_demo_recordings"


_model = None


def setup_whisper():
    global _model

    if _model is not None:
        return _model

    if select_backend() == "cuda":
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"

    _model = faster_whisper.WhisperModel(
        MODEL_NAME,
        device=device,
        compute_type=compute_type,
        cpu_threads=2,
    )
    return _model


def _convert_webm_to_wav(webm_data, work_dir):
    os.makedirs(work_dir, exist_ok=True)

    recording_name = f"user_audio_{uuid.uuid4().hex}"
    webm_path = os.path.join(work_dir, f"{recording_name}.webm")
    wav_path = os.path.join(work_dir, f"{recording_name}.wav")

    with open(webm_path, "wb") as audio_file:
        audio_file.write(webm_data)

    try:
        conversion_result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                webm_path,
                "-ac",
                "1",  # mono
                "-ar",
                "16000",  # 16 kHz
                "-f",
                "wav",  # output format
                "-y",  # overwrite
                wav_path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_S,
        )
        if conversion_result.returncode != 0:
            error_message = conversion_result.stderr.strip()
            Path(wav_path).unlink(missing_ok=True)
            raise RuntimeError(f"Audio conversion failed: {error_message}")
        return os.path.abspath(wav_path)
    except subprocess.TimeoutExpired as error:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Audio conversion timed out after {FFMPEG_TIMEOUT_S} seconds"
        ) from error
    finally:
        Path(webm_path).unlink(missing_ok=True)


def transcribe_bytes(audio_data, work_dir=None):
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    recordings_dir = Path(work_dir) if work_dir is not None else default_work_dir()
    wav_path = _convert_webm_to_wav(audio_data, recordings_dir)
    try:
        segments, _ = _model.transcribe(wav_path, language="en", beam_size=3)
        return " ".join(segment.text.strip() for segment in segments)
    finally:
        Path(wav_path).unlink(missing_ok=True)
