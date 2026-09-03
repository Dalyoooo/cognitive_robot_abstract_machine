
"""Turns recorded audio into text with Whisper.

``transcribe_bytes`` is the original path: one recording in, one transcript out.
The scene functions run Whisper once per speech segment instead, which keeps the
turns apart, gives each one a timestamp, and stops Whisper inventing words over
the silence between them.
"""

import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import faster_whisper

from thesis_demo.audio import vad
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


@dataclass(frozen=True)
class Utterance:
    """One transcribed speech segment out of a recorded scene."""

    text: str
    """The recognised words."""

    start_s: float
    """Offset into the recording where the utterance starts, in seconds."""

    end_s: float
    """Offset where it ends."""

    speaker_id: int | None = None
    """Which voice said it, or None while voices have not been told apart."""


def _run_whisper(audio):
    # ``audio`` is either a file path or a float32 waveform; faster-whisper
    # accepts both. One VAD segment may still be split internally, so join.
    segments, _ = _model.transcribe(audio, language="en", beam_size=3)
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_bytes(audio_data, work_dir=None):
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    recordings_dir = Path(work_dir) if work_dir is not None else default_work_dir()
    wav_path = _convert_webm_to_wav(audio_data, recordings_dir)
    try:
        return _run_whisper(wav_path)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def transcribe_segment(segment):
    """Transcribe a single VAD speech segment into an :class:`Utterance`."""
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    return Utterance(
        text=_run_whisper(segment.samples),
        start_s=segment.start_s,
        end_s=segment.end_s,
    )


def _transcribe_segments(segments, diarizer=None, sampling_rate=vad.SAMPLING_RATE):
    """Transcribe every segment, labelling voices when a diarizer is given.

    Voices are grouped over the untrimmed segment list so clustering sees every
    voice in the scene, including segments whose transcription turns out empty;
    those are dropped afterwards.
    """
    speakers = (
        diarizer.assign(segments) if diarizer is not None else [None] * len(segments)
    )
    utterances = []
    for segment, speaker in zip(segments, speakers):
        utterance = transcribe_segment(segment)
        # Whisper returns nothing for a breath or a laugh that the VAD let
        # through, and an utterance without words has no role to play later.
        if not utterance.text:
            continue
        utterances.append(replace(utterance, speaker_id=speaker))
    return utterances


def transcribe_scene(
    audio, sampling_rate=vad.SAMPLING_RATE, vad_options=None, diarizer=None
):
    """Split a waveform into speech segments and transcribe each one.

    Returns the utterances in time order. Segments whose transcription comes
    back empty (breath, laughter, a stray noise the VAD let through) are
    dropped, so every returned utterance carries actual words. Pass a
    ``diarizer`` to also label which voice said what.
    """
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    segments = vad.detect_segments(audio, sampling_rate, options=vad_options)
    return _transcribe_segments(segments, diarizer, sampling_rate)


def transcribe_scene_bytes(audio_data, work_dir=None, vad_options=None, diarizer=None):
    """Full front-end for a recorded scene: webm bytes to a list of utterances.

    Unlike :func:`transcribe_bytes`, this keeps each speaker turn apart so the
    dialogue stage can weigh them separately against the world context.
    """
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    recordings_dir = Path(work_dir) if work_dir is not None else default_work_dir()
    wav_path = _convert_webm_to_wav(audio_data, recordings_dir)
    try:
        segments = vad.segment_file(wav_path, options=vad_options)
        return _transcribe_segments(segments, diarizer)
    finally:
        Path(wav_path).unlink(missing_ok=True)
