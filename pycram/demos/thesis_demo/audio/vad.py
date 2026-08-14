"""Finds the stretches of a recording that contain speech.

Silero VAD scores short windows of the waveform for speech, and the resulting
timestamps become segments that carry a start time, an end time and the audio
itself. Everything in between, silence and non-vocal noise, is dropped.

Silero ships inside faster-whisper, which this demo already uses to transcribe,
so nothing extra has to be installed for this stage.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys
import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

# Audio is a long row of numbers; 16000 of them make up one second. Both Silero
# and Whisper expect this rate, so every waveform here is at 16 kHz.
SAMPLING_RATE = 16000


def default_options():
    """Voice-activity options tuned for a multi-speaker scene.

    faster-whisper's own defaults (2 s minimum silence) are meant to keep one
    dictation together. A household scene instead needs distinct utterances to
    stay apart, so silences split earlier and a short pad avoids gluing a
    background remark onto the command that precedes it.
    """
    return VadOptions(
        threshold=0.5,  # how sure Silero must be before a window counts as speech
        min_speech_duration_ms=250,  # anything shorter is a click or a cough, not a word
        min_silence_duration_ms=300,  # a gap this long ends the current utterance
        speech_pad_ms=200,  # keep a little audio either side so syllables survive
    )


@dataclass(frozen=True)
class SpeechSegment:
    """One stretch of speech: when it happened and the audio it contains."""

    start_s: float  # offset into the recording, in seconds
    end_s: float
    samples: np.ndarray  # the speech audio itself, float32 at SAMPLING_RATE
    sampling_rate: int = SAMPLING_RATE

    @property
    def duration_s(self):
        return self.end_s - self.start_s


def detect_segments(audio, sampling_rate=SAMPLING_RATE, options=None):
    """Split a waveform into speech segments, dropping the non-speech regions.

    ``audio`` is a float32 waveform in [-1, 1]; the return value is a list of
    :class:`SpeechSegment` in time order, empty when nobody speaks.
    """
    # Silero reads the samples directly, so they have to be float32 and laid out
    # in one contiguous block.
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    options = options or default_options()
    timestamps = get_speech_timestamps(
        audio, options, sampling_rate=sampling_rate
    )

    segments = []
    for span in timestamps:
        # Silero reports positions as sample indices, so dividing by the rate
        # turns them into seconds, and slicing cuts out that piece of audio.
        start_sample, end_sample = span["start"], span["end"]
        segments.append(
            SpeechSegment(
                start_s=start_sample / sampling_rate,
                end_s=end_sample / sampling_rate,
                samples=audio[start_sample:end_sample],
                sampling_rate=sampling_rate,
            )
        )
    return segments


def segment_file(path, options=None):
    """Decode an audio file to float32/16 kHz and split it into speech segments."""
    # decode_audio handles whatever container the file uses and resamples it.
    audio = decode_audio(str(path), sampling_rate=SAMPLING_RATE)
    return detect_segments(audio, sampling_rate=SAMPLING_RATE, options=options)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python -m thesis_demo.audio.vad <audio-file>")
        raise SystemExit(2)

    found = segment_file(sys.argv[1])
    print(f"{len(found)} speech segment(s):")
    for index, segment in enumerate(found):
        print(
            f"  [{index}] {segment.start_s:6.2f}s - {segment.end_s:6.2f}s "
            f"({segment.duration_s:4.2f}s, {len(segment.samples)} samples)"
        )
