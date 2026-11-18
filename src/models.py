"""TTS request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PremasterHealth(BaseModel):
    """Loudness premaster state. It fails OPEN, so without this a broken ffmpeg
    invocation is invisible (PR #64 skipped the premaster on every clip for 2 days)."""

    self_test: Literal["passed", "failed", "disabled", "not_run"]
    reasons: list[str] = Field(
        default_factory=list,
        description="Stable, path-free failure codes (full ffmpeg detail is in the ERROR log only)",
    )
    integrated_lufs: float | None = None
    warning_counts: dict[str, int] = Field(
        default_factory=dict, description="Per-reason premaster warning occurrences since process start"
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    models_loaded: list[str] = Field(default_factory=list)
    provider_engines: dict[str, str] = Field(default_factory=dict)
    word_timings_available: bool = False
    premaster: PremasterHealth | None = None


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    provider: Literal["kokoro", "kitten"] | None = Field(default=None, description="Override default provider")
    voice: str | None = Field(default=None, description="Voice ID (provider-specific)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    sample_rate: int | None = Field(
        default=None,
        ge=8000,
        le=48000,
        description=(
            "Request an output sample rate. This service does NOT resample: the only "
            "accepted value is the engine's own rate, and anything else is rejected "
            "with 400 rather than delivered as a re-stamped header. Omit it to get "
            "whatever the engine produces."
        ),
    )
    # ``mp3`` was advertised here and never produced: the response was WAV bytes with
    # media_type audio/wav and format "wav" — a 200 that contradicted the request.
    # Narrowed so the schema stops promising an encoder that does not exist; pydantic
    # now answers 422 naming the supported value. Restore "mp3" only together with a
    # real encoder. Same policy as video_production_client, which REJECTS an
    # unsupported output_format rather than silently ignoring it.
    output_format: Literal["wav"] = Field(default="wav")
    ssml: bool | None = Field(
        default=None,
        description="Force SSML parsing on (true) / off (false); null = auto-detect from tags",
    )
    master_lufs: float | None = Field(
        default=None,
        ge=-30.0,
        le=-9.0,
        description="Per-clip pre-master integrated-LUFS target; null → −18 default",
    )


class WordTiming(BaseModel):
    """One spoken word and where it sits in the RETURNED audio, in seconds.

    ``start``/``end`` are in the FINAL audio timebase — i.e. offsets into the
    clip the caller just received, after piece concatenation, inter-piece gaps
    and any SSML ``<break>`` silence. Punctuation-only tokens are never emitted
    (the consumer aligns these against authored script WORDS).
    """

    word: str
    start: float
    end: float


class SynthesizeResponse(BaseModel):
    """JSON response with base64 audio (alternative to direct audio/wav stream)."""

    audio_base64: str
    sample_rate: int
    duration_sec: float
    provider: str
    voice: str
    format: str
    word_timings: list[WordTiming] | None = Field(
        default=None,
        description=(
            "Per-word start/end offsets in seconds into the returned audio. Present only when the"
            " Kokoro engine exposes native durations (torch or a timed ONNX graph); absent otherwise, so"
            " existing consumers are unaffected."
        ),
    )
    speech_seconds: float = Field(
        default=0.0,
        description=(
            "duration_sec minus pause_seconds: the portion of the returned audio that is actual"
            " speech. A caller computing a speed correction from TOTAL duration should use THIS"
            " field (or duration_sec - pause_seconds) as the denominator instead — it excludes"
            " silence this service inserted on its own, which does not scale with `speed` and so"
            " should never be divided into a words-per-minute target."
        ),
    )
    pause_seconds: float = Field(
        default=0.0,
        description=(
            "Seconds of the returned audio that are silence this service inserted deliberately:"
            " the inter-piece pacing gap between sentences/clauses, plus any authored SSML <break>"
            " duration. Exact (summed from the samples actually spliced in, never re-measured from"
            " the waveform). Does NOT include kokoro's own per-piece near-silent head/tail padding,"
            " which is part of the model's speech output. 0.0 for a provider (kitten) or an older"
            " code path that inserts no pacing silence — additive field, existing consumers that"
            " ignore it see byte-identical audio/duration_sec/word_timings."
        ),
    )


class VoiceInfo(BaseModel):
    voice_id: str
    provider: str
    language: str | None = None
    gender: str | None = None


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    default_provider: str
    default_voice: str
