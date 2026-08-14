"""Groups speech segments by voice.

Each segment is turned into an embedding, a row of numbers describing the voice
rather than the words. Embeddings of one person land close together, so the
cosine distance between them is small; clustering then puts every segment whose
distance falls under a threshold into the same group and numbers the groups.

The numbers only tell voices apart inside one recording. Nothing here recognises
who anyone is, and no speaker is known in advance.
"""

from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import librosa
import numpy as np
from thesis_demo.audio.vad import SAMPLING_RATE

# Cosine-distance thresholds below which two segments count as the same voice.
# These are starting points, not settled values: the right cut depends on the
# microphone, the room and how much the voices differ. Calibrate them on real
# recordings with ``pairwise_distances`` before trusting a speaker count.
DEFAULT_THRESHOLDS = {"mfcc": 0.15, "ecapa": 0.30}

MIN_EMBEDDING_DURATION_S = 0.3


def mfcc_embedding(samples, sampling_rate=SAMPLING_RATE):
    """Describe a voice by the average shape of its spectrum.

    A dependency-free baseline: mel-frequency cepstral coefficients capture
    timbre, and their mean and spread over the segment form the fingerprint.
    It reacts to loudness and emotion as much as to the person, so it separates
    only clearly different voices -- useful as a comparison baseline, not as a
    strong speaker model.
    """

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    # 20 coefficients per short window, so one row of 20 numbers per window.
    coefficients = librosa.feature.mfcc(
        y=samples, sr=sampling_rate, n_mfcc=20
    )
    # Averaging over all windows gives the typical spectrum of this segment and
    # the spread says how much it varied: 40 numbers, whatever the length was.
    embedding = np.concatenate(
        [coefficients.mean(axis=1), coefficients.std(axis=1)]
    )
    return _normalize(embedding)


def ecapa_model_dir(source):
    """Where the ECAPA weights live locally.

    ``ECAPA_MODEL_DIR`` lets a deployment point at a directory populated when the
    image was built, so a container does not download the model again in every
    session. Without it the weights land in a temporary directory.
    """
    configured = os.environ.get("ECAPA_MODEL_DIR")
    if configured:
        return configured
    return str(Path(tempfile.gettempdir()) / source.replace("/", "_"))


def ecapa_embedding_backend(source="speechbrain/spkrec-ecapa-voxceleb"):
    """Build an ECAPA-TDNN embedding function; needs speechbrain installed.

    The model is trained to describe *who* speaks regardless of *what* is said,
    which is what makes it robust where the MFCC baseline is weak: shouting,
    emotion, and short segments. It runs locally once downloaded.
    """
    # Imported here, not at module level: speechbrain and torch are only needed
    # for this backend, so the MFCC baseline keeps working without them.
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source=source,
        savedir=ecapa_model_dir(source),
    )

    def embed(samples, sampling_rate=SAMPLING_RATE):

        # The model was trained on 16 kHz audio and takes the waveform as-is, so
        # samples at another rate would be read at the wrong speed and yield a
        # fingerprint of nobody. Refuse rather than return something plausible.
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
        return _normalize(embedding)

    return embed


def _normalize(vector):
    """Scale a vector to length 1 so only its direction carries information."""
    vector = np.asarray(vector, dtype=np.float64).ravel()
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def cosine_distance(first, second):
    """0 for identical directions, 1 for unrelated, 2 for opposite ones."""
    # Both vectors are already length 1, so their dot product is the cosine of
    # the angle between them and subtracting it from 1 turns it into a distance.
    return float(1.0 - np.dot(first, second))


def pairwise_distances(segments, embed=None):
    """Cosine distance between every pair of segments.

    Made public because choosing a threshold is an empirical step: printing this
    matrix for a recording shows where same-voice distances end and
    different-voice distances begin.
    """
    embed = embed or mfcc_embedding
    embeddings = [embed(segment.samples, segment.sampling_rate) for segment in segments]
    count = len(embeddings)
    matrix = np.zeros((count, count))
    for row in range(count):
        for column in range(row + 1, count):
            distance = cosine_distance(embeddings[row], embeddings[column])
            matrix[row][column] = matrix[column][row] = distance
    return matrix


@dataclass
class Diarizer:
    """Groups speech segments by voice, without knowing who the speakers are.

    Speaker numbers are labels for telling voices apart, not identities: the
    same person keeps one number within a recording, and nothing is claimed
    about who they are. The number of speakers is not fixed in advance -- the
    threshold decides it, which is what a household scene needs.
    """

    embed: object = None
    threshold: float = None
    backend_name: str = "mfcc"
    # Set by assign() to the indices it could not judge, so a caller can see
    # which voices were left unknown rather than having to infer it.
    skipped_short: list = field(default_factory=list)

    def __post_init__(self):
        if self.embed is None:
            self.embed = mfcc_embedding
        if self.threshold is None:
            self.threshold = DEFAULT_THRESHOLDS.get(self.backend_name, 0.15)

    def assign(self, segments):
        """Return one speaker label per segment, numbered from 1.

        Segments too short for a reliable fingerprint get None rather than a
        guessed label, so downstream counting can see that the voice is unknown
        instead of trusting a coin flip.
        """
        if not segments:
            return []

        self.skipped_short = []
        # usable holds the positions in `segments` that are long enough to judge,
        # and embeddings holds one vector for each of them, in the same order.
        usable, embeddings = [], []
        for index, segment in enumerate(segments):
            if segment.duration_s < MIN_EMBEDDING_DURATION_S:
                self.skipped_short.append(index)
                continue
            usable.append(index)
            # Each segment carries the rate its samples were captured at, and a
            # fingerprint computed against the wrong rate describes no one.
            embeddings.append(self.embed(segment.samples, segment.sampling_rate))

        labels = [None] * len(segments)
        if not usable:
            return labels
        if len(usable) == 1:
            # Nothing to compare against, so the one voice present is speaker 1.
            labels[usable[0]] = 1
            return labels

        # One clustering run covers every segment at once; the loop only copies
        # each result back to the position the segment came from.
        clustered = self._cluster(np.vstack(embeddings))
        for position, index in enumerate(usable):
            labels[index] = int(clustered[position])
        return labels

    def _cluster(self, embeddings):
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(
            # n_clusters=None with a distance_threshold means the number of
            # speakers comes out of the data instead of being fixed beforehand:
            # groups keep merging while they are closer than the threshold.
            n_clusters=None,
            distance_threshold=self.threshold,
            metric="cosine",
            linkage="average",
        ).fit(embeddings)
        # sklearn hands back arbitrary group ids, so they are renumbered by first
        # appearance: the voice heard first becomes speaker 1, the next one 2.
        order, renumbered = {}, []
        for label in clustering.labels_:
            if label not in order:
                order[label] = len(order) + 1
            renumbered.append(order[label])
        return renumbered


def load_diarizer(backend="mfcc", threshold=None):
    """Build a :class:`Diarizer` for the named embedding backend."""
    if backend == "mfcc":
        return Diarizer(embed=mfcc_embedding, threshold=threshold, backend_name="mfcc")
    if backend == "ecapa":
        return Diarizer(
            embed=ecapa_embedding_backend(),
            threshold=threshold,
            backend_name="ecapa",
        )
    raise ValueError(f"Unknown backend {backend!r}; choose 'mfcc' or 'ecapa'")
