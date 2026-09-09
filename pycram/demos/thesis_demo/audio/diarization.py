"""Groups speech segments by voice.

Each segment becomes an embedding, a row of numbers describing the voice rather
than the words. Embeddings of one person lie close together, so clustering puts
every segment whose cosine distance falls under a threshold into one group and
numbers the groups.

.. note:: The numbers only tell voices apart within one recording. Nothing here
    recognises who anyone is, and no speaker is known in advance.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import librosa
import numpy as np

from thesis_demo.audio.vad import SAMPLING_RATE, SpeechSegment, segment_file

# %% configuration


class EmbeddingBackend(StrEnum):
    """Which model describes a voice."""

    MFCC = "mfcc"
    ECAPA = "ecapa"


class LinkageMethod(StrEnum):
    """How the distance between two groups of embeddings is measured."""

    AVERAGE = "average"


class DistanceMetric(StrEnum):
    """How the distance between two embeddings is measured."""

    COSINE = "cosine"


ECAPA_MODEL_DIR_VARIABLE = "ECAPA_MODEL_DIR"
"""Environment variable naming a directory the ECAPA weights already live in."""

THRESHOLD_VARIABLE = "VAD_VOICE_CUTOFF"
"""Environment variable overriding the cosine cutoff for every backend.

The right cut is a measurement, not a constant, and finding it means trying
values against the same recordings. Reading it from the environment keeps that
a one-line change instead of an edit plus a rebuild of the lab image.
"""

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
"""Model the ECAPA backend loads."""

DEFAULT_THRESHOLDS = {
    EmbeddingBackend.MFCC: 0.15,
    EmbeddingBackend.ECAPA: 0.72,
}
"""Cosine distance below which two segments count as the same voice.

One voice measures much wider through the lab's browser recorder than through a
local microphone, because the voice reaches the pipeline as Opus rather than as
it was spoken. Measured between segments of a single speaker: 0.22, 0.23, 0.31
and 0.38 locally; 0.59 to 0.74 through the browser while it still cancelled echo
and rode the gain; 0.48 to 0.64 once that was switched off. Between two
speakers, through the same browser channel: 0.84 and 0.86.

The cut sits at 0.72, near the middle of what is left between those two groups.
Every earlier value was inside the same-voice range rather than above it: 0.30
and 0.50 each split one speaker into three, and 0.65 cleared the widest
same-voice pair by 0.01, which is luck and not a margin.

Length moves the distance as much as the person does. The widest pair a single
speaker produced, 0.64, was between his two shortest and most hesitant
fragments, while his two full sentences sat at 0.49. A 0.78 s filler reached
0.74. That is what :data:`MIN_ANCHOR_DURATION_S` answers, so the cut does not
have to absorb it.

.. warning:: A handful of measurements from a few recordings is a starting
    point, not a calibration. The right cut depends on the microphone, the room
    and how much the voices differ, so run this module over real recordings
    (see its command line) and put the cut where the two groups separate.
"""

FALLBACK_THRESHOLD = 0.15
"""Threshold for a backend :data:`DEFAULT_THRESHOLDS` does not cover."""

MIN_EMBEDDING_DURATION_S = 0.3
"""Shortest segment a voice can be judged from."""

MIN_ANCHOR_DURATION_S = 0.7
"""Shortest segment that may found a voice of its own.

The embedding of a sub-second snippet describes the snippet nearly as much as
the person, so a filler like "Um" lands far from the same speaker's sentences.
Such a segment may still join the voice it is closest to, but it can no longer
start a new one and pull a scene apart.

At 1.0 s the bar sat just above the last line of a scene: a closing remark runs
0.74 to 1.0 s, could therefore found no voice of its own, and -- being a speaker
who says nothing else -- was close to no other voice either, so it came back
unknown. Measured over 20 recorded scenes with four speakers each, 1.0 left 9
utterances without a voice and named every speaker correctly in 11 scenes; 0.7
leaves none unknown and gets 18. Below 0.7 nothing improves further, so the cut
sits at the point the measurements stop moving rather than at the edge of them.
"""

MFCC_COEFFICIENT_COUNT = 20
"""Coefficients per analysis window in the MFCC baseline."""

FIRST_SPEAKER_LABEL = 1
"""Number given to the first voice heard."""


# %% embedding backends


def mfcc_embedding(samples, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
    """Describe a voice by the average shape of its spectrum.

    .. note:: This baseline needs no extra packages, but it reacts to loudness
        and emotion as much as to the person, so it separates only clearly
        different voices.
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    coefficients = librosa.feature.mfcc(
        y=samples, sr=sampling_rate, n_mfcc=MFCC_COEFFICIENT_COUNT
    )
    # The mean over all windows is the typical spectrum, the spread says how much
    # it varied: a fixed-length description whatever the segment length was.
    embedding = np.concatenate(
        [coefficients.mean(axis=1), coefficients.std(axis=1)]
    )
    return normalize(embedding)


def ecapa_embedding_backend(source: str = ECAPA_SOURCE):
    """Build an ECAPA-TDNN embedding function.

    The model describes *who* speaks regardless of *what* is said, which holds up
    where the MFCC baseline is weakest: shouting, emotion and short segments. It
    runs locally once downloaded.
    """
    # Imported here rather than at module level so the MFCC baseline keeps working
    # where speechbrain and torch are not installed.
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source=source, savedir=ecapa_model_dir(source)
    )

    def embed(samples, sampling_rate: int = SAMPLING_RATE) -> np.ndarray:
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(
                f"ECAPA expects {SAMPLING_RATE} Hz audio, got {sampling_rate} Hz; "
                "resample the segment before embedding it"
            )
        waveform = torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            embedding = encoder.encode_batch(waveform).squeeze().cpu().numpy()
        return normalize(embedding)

    return embed


def ecapa_model_dir(source: str = ECAPA_SOURCE) -> str:
    """Return the directory the ECAPA weights are read from and written to.

    :data:`ECAPA_MODEL_DIR_VARIABLE` names it where a deployment pre-loaded the
    model; otherwise a temporary directory is used.
    """
    configured = os.environ.get(ECAPA_MODEL_DIR_VARIABLE)
    if configured:
        return configured
    return str(Path(tempfile.gettempdir()) / source.replace("/", "_"))


# %% distances


def normalize(vector) -> np.ndarray:
    """Scale a vector to length 1 so only its direction carries information."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_distance(first, second) -> float:
    """Return 0 for identical directions, 1 for unrelated, 2 for opposite ones."""
    # Both vectors have length 1, so their dot product is the cosine of the angle
    # between them.
    return float(1.0 - np.dot(first, second))


def _distance_matrix(embeddings) -> np.ndarray:
    """Cosine distance between every pair of already-computed embeddings."""
    count = len(embeddings)
    matrix = np.zeros((count, count))
    for row in range(count):
        for column in range(row + 1, count):
            distance = cosine_distance(embeddings[row], embeddings[column])
            matrix[row][column] = matrix[column][row] = distance
    return matrix


def pairwise_distances(segments: list[SpeechSegment], embed=None) -> np.ndarray:
    """Return the cosine distance between every pair of segments.

    Printing this for a recording shows where same-voice distances end and
    different-voice distances begin, which is how a threshold is chosen.
    """
    embed = embed or mfcc_embedding
    embeddings = [
        embed(segment.samples, segment.sampling_rate) for segment in segments
    ]
    return _distance_matrix(embeddings)


# %% grouping


@dataclass
class Diarizer:
    """Groups speech segments by voice, without knowing who the speakers are.

    The number of voices is not fixed in advance; :attr:`threshold` decides it.
    """

    embed: object = None
    """Function mapping samples and their rate to an embedding."""

    threshold: float = None
    """Cosine distance below which two segments count as one voice."""

    backend_name: EmbeddingBackend = EmbeddingBackend.MFCC
    """Which backend :attr:`embed` came from, used to pick a default threshold."""

    skipped_short: list[int] = field(default_factory=list)
    """Indices the last :meth:`assign` could not judge, being too short."""

    last_distances: np.ndarray | None = field(default=None)
    """Distances between the segments the last :meth:`assign` judged.

    The threshold decides where one voice ends and the next begins, and the
    right cut depends on the microphone, the room and the speakers. Keeping the
    measured distances lets a caller see how far the recording sat from the cut
    that was applied, instead of only the verdict it produced.
    """

    last_judged: list[int] = field(default_factory=list)
    """Which segment indices those distances belong to, in the same order."""

    attached: list[int] = field(default_factory=list)
    """Indices the last :meth:`assign` joined to a voice instead of clustering.

    These are the segments too short to found a voice. Their label was decided
    by proximity to a longer one, which is a weaker claim than a clustered
    label, so a caller can say so rather than presenting both alike.
    """

    def __post_init__(self):
        if self.embed is None:
            self.embed = mfcc_embedding
        if self.threshold is None:
            self.threshold = DEFAULT_THRESHOLDS.get(
                self.backend_name, FALLBACK_THRESHOLD
            )

    def assign(self, segments: list[SpeechSegment]) -> list[int | None]:
        """Return one speaker label per segment, numbered in order of appearance.

        Segments long enough to found a voice are clustered; shorter ones join
        the voice they are closest to, so a filler cannot become a speaker.

        .. note:: A segment shorter than :data:`MIN_EMBEDDING_DURATION_S`, or one
            close to no voice at all, gets None rather than a guessed label, so
            a caller can see that the voice is unknown.
        """
        if not segments:
            return []

        self.skipped_short = []
        self.attached = []
        self.last_distances = None
        self.last_judged = []
        usable, embeddings = [], []
        for index, segment in enumerate(segments):
            if segment.duration_s < MIN_EMBEDDING_DURATION_S:
                self.skipped_short.append(index)
                continue
            usable.append(index)
            # The rate comes from the segment: an embedding computed against the
            # wrong rate describes no one.
            embeddings.append(self.embed(segment.samples, segment.sampling_rate))

        labels: list[int | None] = [None] * len(segments)
        if not usable:
            return labels
        if len(usable) == 1:
            labels[usable[0]] = FIRST_SPEAKER_LABEL
            return labels

        stacked = np.vstack(embeddings)
        self.last_distances = _distance_matrix(stacked)
        self.last_judged = list(usable)

        # Positions within `usable`, not indices into `segments`.
        anchors = [
            position
            for position, index in enumerate(usable)
            if segments[index].duration_s >= MIN_ANCHOR_DURATION_S
        ]
        # Nothing long enough to anchor: the short segments are all the recording
        # holds, so they are judged against each other rather than left unknown.
        if not anchors:
            anchors = list(range(len(usable)))
        anchored = set(anchors)
        joiners = [
            position for position in range(len(usable)) if position not in anchored
        ]

        # One run covers every anchor; the loop only copies each result back to
        # the position its segment came from.
        voices = (
            [FIRST_SPEAKER_LABEL]
            if len(anchors) == 1
            else self._cluster(stacked[anchors])
        )
        for position, voice in zip(anchors, voices):
            labels[usable[position]] = int(voice)

        for position in joiners:
            voice = self._closest_voice(position, anchors, voices)
            if voice is not None:
                self.attached.append(usable[position])
            labels[usable[position]] = voice
        return labels

    def _closest_voice(self, position, anchors, voices) -> int | None:
        """Return the voice a short segment belongs to, or None if none is close.

        The distance to a voice is the mean distance to its anchors, matching the
        average linkage the clustering itself uses.
        """
        distances_per_voice: dict[int, list[float]] = {}
        for anchor, voice in zip(anchors, voices):
            distances_per_voice.setdefault(int(voice), []).append(
                float(self.last_distances[position][anchor])
            )
        voice, mean = min(
            (
                (voice, sum(distances) / len(distances))
                for voice, distances in distances_per_voice.items()
            ),
            key=lambda candidate: candidate[1],
        )
        # Joining the nearest voice whatever the distance would give every filler
        # a speaker; beyond the cutoff the honest answer is that it is unknown.
        return voice if mean < self.threshold else None

    def _cluster(self, embeddings) -> list[int]:
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            # Without a fixed cluster count, groups keep merging while they are
            # closer than the threshold, so the data decides how many voices.
            n_clusters=None,
            distance_threshold=self.threshold,
            metric=DistanceMetric.COSINE.value,
            linkage=LinkageMethod.AVERAGE.value,
        ).fit(embeddings)
        # The group ids are arbitrary, so they are renumbered by first appearance.
        order, renumbered = {}, []
        for label in clustering.labels_:
            if label not in order:
                order[label] = len(order) + FIRST_SPEAKER_LABEL
            renumbered.append(order[label])
        return renumbered


def configured_threshold() -> float | None:
    """Return the cutoff :data:`THRESHOLD_VARIABLE` asks for, or None.

    :raises ValueError: when the variable is set but not a usable distance.
    """
    # A variable set to nothing is how a shell says "leave it alone", so an
    # empty or blank setting means the backend default rather than an error.
    raw = os.environ.get(THRESHOLD_VARIABLE, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"{THRESHOLD_VARIABLE}={raw!r} is not a number; a cosine distance "
            "between 0 and 2 was expected"
        ) from None
    if not 0.0 < value < 2.0:
        raise ValueError(
            f"{THRESHOLD_VARIABLE}={value} lies outside the range a cosine "
            "distance can take; 0 would keep nothing together and 2 everything"
        )
    return value


def load_diarizer(
    backend: EmbeddingBackend = EmbeddingBackend.MFCC, threshold: float = None
) -> Diarizer:
    """Build a :class:`Diarizer` for the named embedding backend.

    :raises ValueError: for a backend that does not exist.
    """
    backend = EmbeddingBackend(backend)
    if threshold is None:
        threshold = configured_threshold()
    if backend is EmbeddingBackend.MFCC:
        return Diarizer(
            embed=mfcc_embedding, threshold=threshold, backend_name=backend
        )
    return Diarizer(
        embed=ecapa_embedding_backend(), threshold=threshold, backend_name=backend
    )


# %% command line


def _print_measurements(path: str, backend: EmbeddingBackend) -> None:
    """Print what a recording measured, which is how a cutoff gets chosen.

    The verdict alone cannot be argued with. The durations say which segments
    were allowed to found a voice, and the matrix says how far apart the
    recording actually sat from the cutoff that was applied to it.
    """
    diarizer = load_diarizer(backend)
    segments = segment_file(path)
    labels = diarizer.assign(segments)

    print(f"{len(segments)} speech segment(s), {backend.value} embeddings, "
          f"cutoff {diarizer.threshold:.2f}:")
    for index, segment in enumerate(segments):
        label = labels[index]
        voice = f"voice {label}" if label is not None else "voice unknown"
        if index in diarizer.attached:
            voice += " (short, joined)"
        elif index in diarizer.skipped_short:
            voice += " (too short to judge)"
        print(
            f"  [{index}] {segment.start_s:6.2f}s - {segment.end_s:6.2f}s "
            f"({segment.duration_s:4.2f}s)  {voice}"
        )

    judged = diarizer.last_judged
    if diarizer.last_distances is None or len(judged) < 2:
        print("\nnot enough judged segments to measure a distance")
        return

    print("\ncosine distance between judged segments:")
    print("       " + "".join(f"[{index:>3}]" for index in judged))
    for row, index in enumerate(judged):
        cells = "".join(
            f"{diarizer.last_distances[row][column]:>5.2f}"
            for column in range(len(judged))
        )
        print(f"  [{index:>3}]{cells}")


if __name__ == "__main__":

    if not 2 <= len(sys.argv) <= 3:
        print(
            "usage: python -m thesis_demo.audio.diarization <audio-file> "
            "[mfcc|ecapa]"
        )
        raise SystemExit(2)

    chosen = (
        EmbeddingBackend(sys.argv[2]) if len(sys.argv) == 3 else EmbeddingBackend.MFCC
    )
    _print_measurements(sys.argv[1], chosen)
