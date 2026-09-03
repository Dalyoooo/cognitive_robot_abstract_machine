"""Runs the whole speech front-end in one call.

Audio goes in, one instruction for the planner comes out, via segmentation,
transcription, optional grouping by voice, and the triage that decides what each
utterance is worth.
"""

from __future__ import annotations

from thesis_demo.dialogue.interpreter import interpret


def understand_scene(
    audio_bytes,
    context,
    generate,
    work_dir=None,
    vad_options=None,
    diarizer=None,
    max_attempts=2,
):
    """Turn one recorded scene into a single instruction for the planner.

    Runs the whole speech front-end: voice-activity segmentation, per-segment
    transcription, optional grouping of segments by voice, then the
    context-aware interpretation that keeps the instruction, folds relevant
    background remarks into it and drops noise. Returns the list of transcribed
    utterances together with the :class:`InterpretationResult`; the result's
    ``instruction`` is what feeds ``plan()``.

    Passing a ``diarizer`` (see :mod:`thesis_demo.audio.diarization`) labels who
    said what, which is what lets repeated identical claims from one person be
    counted once instead of several times.

    ``transcribe_scene_bytes`` is imported lazily so this module stays importable
    where faster-whisper is absent (e.g. the planner-only environment).
    """
    from thesis_demo.audio.transcriber import transcribe_scene_bytes

    utterances = transcribe_scene_bytes(
        audio_bytes,
        work_dir=work_dir,
        vad_options=vad_options,
        diarizer=diarizer,
    )
    result = interpret(utterances, context, generate, max_attempts=max_attempts)
    return utterances, result
