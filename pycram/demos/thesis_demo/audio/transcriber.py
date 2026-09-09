
"""Turns recorded audio into text with Whisper.

``transcribe_bytes`` is the original path: one recording in, one transcript out.
The scene functions run Whisper once per speech segment instead, which keeps the
turns apart, gives each one a timestamp, and stops Whisper inventing words over
the silence between them.

Whisper also reports where each sentence sits inside the segment it was given,
and a segment can hold more than one speaker whenever they left the VAD too
little silence to separate them. Those spans are therefore kept rather than
joined, so a scene comes out as one utterance per sentence, each with its own
voice -- which is what lets the dialogue stage give each sentence its own role.
"""

import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Sentence:
    """One sentence heard inside a speech segment, with the audio it came from.

    Keeping the audio beside the words is what lets each sentence be judged for
    its own voice, instead of inheriting the voice of the whole segment it was
    cut out of.
    """

    segment: vad.SpeechSegment
    """The stretch of audio this sentence was heard in."""

    text: str
    """The recognised words."""


def _whisper_spans(audio):
    # ``audio`` is either a file path or a float32 waveform; faster-whisper
    # accepts both. Each span it reports covers one sentence it heard, timed
    # from the start of the audio it was given.
    spans, _ = _model.transcribe(audio, language="en", beam_size=3)
    return list(spans)


def _run_whisper(audio):
    """Transcribe one recording into a single line of text."""
    return " ".join(span.text.strip() for span in _whisper_spans(audio)).strip()


def transcribe_bytes(audio_data, work_dir=None):
    if _model is None:
        raise RuntimeError("Whisper model not loaded. Call setup_whisper() first.")
    recordings_dir = Path(work_dir) if work_dir is not None else default_work_dir()
    wav_path = _convert_webm_to_wav(audio_data, recordings_dir)
    try:
        return _run_whisper(wav_path)
    finally:
        Path(wav_path).unlink(missing_ok=True)


def split_into_sentences(segment, transcribe=None):
    """Split one speech segment into the sentences that were heard inside it.

    Two speakers who leave the VAD too little silence to separate them arrive as
    a single segment, but Whisper still reports a span per sentence within it, so
    the turns come apart on those spans. ``transcribe`` maps a waveform to those
    spans and defaults to Whisper itself.
    """
    transcribe = transcribe or _whisper_spans
    return [
        Sentence(
            segment=segment.cut_between(span.start, span.end), text=span.text.strip()
        )
        for span in transcribe(segment.samples)
    ]


def transcribe_segments(segments, diarizer=None, transcribe=None):
    """Turn speech segments into one utterance per sentence, in time order.

    Each segment is split into its sentences first, so a voice is judged per
    sentence rather than per segment. Sentences that came back without words are
    dropped before the grouping runs, which also keeps the indices a diarizer
    reports -- its short-and-joined bookkeeping -- lined up with the utterances
    returned here.
    """
    sentences = [
        sentence
        for segment in segments
        for sentence in split_into_sentences(segment, transcribe)
        # Whisper returns nothing for a breath or a laugh that the VAD let
        # through, and an utterance without words has no role to play later.
        if sentence.text
    ]
    speakers = (
        diarizer.assign([sentence.segment for sentence in sentences])
        if diarizer is not None
        else [None] * len(sentences)
    )
    return [
        Utterance(
            text=sentence.text,
            start_s=sentence.segment.start_s,
            end_s=sentence.segment.end_s,
            speaker_id=speaker,
        )
        for sentence, speaker in zip(sentences, speakers)
    ]


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
    return transcribe_segments(segments, diarizer)


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
        return transcribe_segments(segments, diarizer)
    finally:
        Path(wav_path).unlink(missing_ok=True)
