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

SPEECH_PROBABILITY_THRESHOLD = 0.70
"""How sure Silero must be before a window counts as speech.

Recorded scenes sit at -20 to -29 dBFS over a noise floor around -44 dB, close
enough that at 0.5 the room itself keeps a segment open across the gap between
two speakers. Raising the bar makes the gap register as the silence it is.
"""

MIN_SPEECH_DURATION_MS = 400
"""Shortest stretch that counts as speech rather than a click or a cough.

At 250 ms one recording produced a 0.36 s region that transcribed to nothing:
long enough to be kept, too short to hold a word.
"""

MIN_SILENCE_DURATION_MS = 100
"""A gap this long ends the current utterance.

This is the number that decides whether two speakers arrive as one region. In
conversation the next person starts 100-300 ms after the last one stops, so the
2000 ms faster-whisper defaults to -- and the 300 ms used here before -- merge
turns that the stages downstream then cannot tell apart. Measured over 20
recorded scenes, 300 ms cut as many regions as the script had utterances in 2
scenes out of 20; 100 ms does so in 16.
"""

SPEECH_PAD_MS = 50
"""How much audio either side of a region is kept, so syllables survive.

Silero splits a gap shorter than twice this pad down the middle, so a pad at or
above half of :data:`MIN_SILENCE_DURATION_MS` spends the whole silence and
leaves one turn's boundary inside the next turn's audio.
"""


def default_options():
    """Voice-activity options tuned for a multi-speaker scene.

    faster-whisper's own defaults (2 s minimum silence) are meant to keep one
    dictation together. A household scene instead needs distinct utterances to
    stay apart, so silences split earlier and the pad stays small enough not to
    glue a background remark onto the command that precedes it.

    .. note:: A gap the VAD cannot see at all -- two speakers running into each
        other -- is not recoverable here. That case is handled a stage later, by
        splitting a region on the sentences Whisper reports inside it.
    """
    return VadOptions(
        threshold=SPEECH_PROBABILITY_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )


@dataclass(frozen=True)
class SpeechSegment:
    """One stretch of speech: when it happened and the audio it contains."""

    start_s: float
    """Offset into the recording where the speech starts, in seconds."""

    end_s: float
    """Offset where it ends."""

    samples: np.ndarray
    """The speech audio itself, float32 at :attr:`sampling_rate`."""

    sampling_rate: int = SAMPLING_RATE
    """Samples per second the audio was captured at."""

    @property
    def duration_s(self) -> float:
        """How long the segment lasts, in seconds."""
        return self.end_s - self.start_s

    def cut_between(self, start_offset_s: float, end_offset_s: float) -> SpeechSegment:
        """Return the part of this segment between two offsets from its start.

        The offsets are measured from the start of this segment, which is how
        Whisper reports the sentences it finds inside one, and they are held
        within the segment so a span reaching past the audio cannot invent any.
        """
        start_offset_s = max(0.0, start_offset_s)
        end_offset_s = min(end_offset_s, self.duration_s)
        # An end at or before the start means the words came with no audio to
        # judge them by; the piece is kept empty so the words are not lost, and
        # a segment this short is left unjudged by the grouping stage.
        end_offset_s = max(end_offset_s, start_offset_s)
        return SpeechSegment(
            start_s=self.start_s + start_offset_s,
            end_s=self.start_s + end_offset_s,
            samples=self.samples[
                int(start_offset_s * self.sampling_rate) : int(
                    end_offset_s * self.sampling_rate
                )
            ],
            sampling_rate=self.sampling_rate,
        )


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
