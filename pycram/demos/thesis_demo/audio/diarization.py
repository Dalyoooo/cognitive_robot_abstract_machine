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
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import librosa
import numpy as np

from thesis_demo.audio.vad import SAMPLING_RATE, SpeechSegment

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

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
"""Model the ECAPA backend loads."""

DEFAULT_THRESHOLDS = {
    EmbeddingBackend.MFCC: 0.15,
    EmbeddingBackend.ECAPA: 0.30,
}
"""Cosine distance below which two segments count as the same voice.

.. warning:: Starting points only. The right cut depends on the microphone, the
    room and how much the voices differ, so calibrate with
    :func:`pairwise_distances` on real recordings before trusting a speaker count.
"""

FALLBACK_THRESHOLD = 0.15
"""Threshold for a backend :data:`DEFAULT_THRESHOLDS` does not cover."""

MIN_EMBEDDING_DURATION_S = 0.3
"""Shortest segment a voice can be judged from."""

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
    count = len(embeddings)
    matrix = np.zeros((count, count))
    for row in range(count):
        for column in range(row + 1, count):
            distance = cosine_distance(embeddings[row], embeddings[column])
            matrix[row][column] = matrix[column][row] = distance
    return matrix


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

    def __post_init__(self):
        if self.embed is None:
            self.embed = mfcc_embedding
        if self.threshold is None:
            self.threshold = DEFAULT_THRESHOLDS.get(
                self.backend_name, FALLBACK_THRESHOLD
            )

    def assign(self, segments: list[SpeechSegment]) -> list[int | None]:
        """Return one speaker label per segment, numbered in order of appearance.

        .. note:: A segment shorter than :data:`MIN_EMBEDDING_DURATION_S` gets
            None rather than a guessed label, so a caller can see that the voice
            is unknown.
        """
        if not segments:
            return []

        self.skipped_short = []
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

        # One run covers every segment; the loop only copies each result back to
        # the position its segment came from.
        clustered = self._cluster(stacked)
        for position, index in enumerate(usable):
            labels[index] = int(clustered[position])
        return labels

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


def load_diarizer(
    backend: EmbeddingBackend = EmbeddingBackend.MFCC, threshold: float = None
) -> Diarizer:
    """Build a :class:`Diarizer` for the named embedding backend.

    :raises ValueError: for a backend that does not exist.
    """
    backend = EmbeddingBackend(backend)
    if backend is EmbeddingBackend.MFCC:
        return Diarizer(
            embed=mfcc_embedding, threshold=threshold, backend_name=backend
        )
    return Diarizer(
        embed=ecapa_embedding_backend(), threshold=threshold, backend_name=backend
    )
