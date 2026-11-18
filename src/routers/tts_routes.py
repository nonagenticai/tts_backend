"""TTS API routes — synthesize + list voices.

Note: a previously-existing GET ``/api/tts/health`` route was dropped because
it shadowed the canonical ``/health`` endpoint at the app level and confused
both consumers and the readiness probe.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import threading
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter
from w2v_common.admission import GateClosedError
from w2v_common.auth import LICENSE_GATE
from w2v_common.auth.identity_context import (
    get_license_id_from_context,
    get_stage_from_context,
    get_subject_id_from_context,
)

from auth.dependencies import get_current_user, resolve_app_id
from auth.models import User
from models import (
    SynthesizeRequest,
    SynthesizeResponse,
    VoiceInfo,
    VoicesResponse,
    WordTiming,
)
from routers.responses import AUTH_RESPONSES, TTS_ENDPOINT_RESPONSES
from services.admission import SYNTH_LANE, get_gpu_gate
from services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    funding_for_provider,
    schedule_cost_event,
    synthesis_event_key,
)
from services.post_processor import premaster_float_wav
from services.tts_service import (
    ProviderUnavailableError,
    SynthesisResult,
    TTSService,
    encode_wav,
)
from services.tts_service import (
    WordTiming as ServiceWordTiming,
)

logger = logging.getLogger(__name__)

# Cancelled synthesis. A client that hangs up (or an orchestrator that times out)
# mid-`/synthesize` cancels the awaiting task, the admission gate's `run` re-raises
# CancelledError at the await point, and the function unwinds BEFORE the
# `schedule_cost_event` below. The worker thread is NOT cancelled -- it keeps
# running to completion on the GPU (holding its lane slot) -- so the work is fully
# paid for in wall time and emits NOTHING. (A request cancelled while still
# QUEUED never starts and is not counted.) That makes the loss invisible in
# exactly the two places anyone would look: billing sees no event, and the
# access log sees no response.
#
# This counter is the measurement, and it is DELIBERATELY the whole change:
# nothing here shields the emit or reorders it. The emit-after-synthesis
# ordering is correct as it stands -- a cost event for audio that was never
# produced would be a worse defect than the gap it closes -- so the first step
# is to find out how often this actually happens, not to pre-emptively bill for
# it. `raise` is unconditional: swallowing CancelledError would break
# cooperative cancellation and strand the caller.
#
# Registered on the default prometheus_client REGISTRY, which is what
# `Instrumentator(...).expose(app, endpoint="/metrics")` in main.py serves, so
# it needs no wiring beyond this declaration.
TTS_CANCELLED = Counter(
    "tts_synthesis_cancelled_total",
    "Synthesis requests cancelled by the caller while the worker thread was still running. "
    "The GPU work completed and was never metered.",
)
# Entitlement gate, mounted router-wide. ``AuthMiddleware`` answers "who are
# you" and stops there: a tenant whose licence is expired or revoked still
# presents a syntactically valid token, and every synthesis it asked for was
# served AND billed (1,741 `w2v-tts` rows in `billing_cost_events`). LICENSE_GATE
# is the "may you". It ships DORMANT — `require_license` is a no-op until
# `W2V_LICENSE_GATE_ENABLED` is true in this service's env — so merging this
# changes nothing until gitops arms the flag, and it never applies to the
# platform's probe paths (`w2v_common.auth.is_license_exempt_path`), which is why
# there is no hand-rolled exemption list here to drift and 402 the health check.
router = APIRouter(prefix="/api/tts", tags=["TTS"], dependencies=list(LICENSE_GATE))


def _word_timings_in_output_timebase(
    timings: list[ServiceWordTiming], engine_sr: int, target_sr: int
) -> list[WordTiming] | None:
    """Map engine-timebase word offsets onto the WAV the caller actually receives.

    Returns ``None`` (→ field omitted) when the engine produced no timings, which is
    the default kokoro-onnx path.

    Two length questions had to be settled before these seconds mean anything:

    1. ``premaster`` — measured, not assumed. It re-encodes with ``-ar <INPUT sample
       rate>`` (``post_processor._sample_rate`` reads the rate off the input file),
       so it never resamples, and ``linear=true`` loudnorm is pure gain (ffmpeg can
       silently fall back to *dynamic* gain when the linear one would breach the true
       peak — ``post_processor._check_normalization_type`` now WARNs when it does;
       either way the frame count is preserved, so the timings below hold). Verified on
       ffmpeg 4.4 over 1 s / 4.4 s / 12 s clips: frame count and sample rate come
       back IDENTICAL, delta 0 frames. Timings survive it untouched.

    2. ``encode_wav`` with a ``sample_rate`` OVERRIDE — this one does move things.
       ``soundfile.write`` does NOT resample; it stamps the SAME samples with the
       requested rate in the header, so a request asking for 48 kHz out of a 24 kHz
       engine gets a clip that plays twice as fast and half as long. (Pre-existing
       behaviour, untouched here — note ``duration_sec`` in this same response is
       already wrong in that case for the same reason.) Since the sample COUNT is
       what is preserved, the honest mapping onto the returned file is a linear
       rescale by ``engine_sr / target_sr``, which is exact.
    """
    if not timings:
        return None
    scale = (engine_sr / target_sr) if (target_sr and target_sr != engine_sr) else 1.0
    return [WordTiming(word=w.word, start=w.start * scale, end=w.end * scale) for w in timings]


def _service_from_request(request: Request) -> TTSService:
    svc = getattr(request.app.state, "tts_service", None)
    if svc is None or not svc.loaded_models():
        raise HTTPException(status_code=503, detail="TTS service has no loaded providers")
    return svc


@router.post(
    "/synthesize",
    operation_id="tts_synthesize",
    summary="Synthesize speech (returns audio/wav or JSON+base64 when Accept: application/json)",
    responses=TTS_ENDPOINT_RESPONSES,
)
async def synthesize(
    request: Request,
    body: SynthesizeRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> Response:
    """Synthesize speech and return audio/wav bytes (or JSON+base64 if Accept: application/json)."""
    svc = _service_from_request(request)
    started = time.monotonic()
    # Set by the worker thread the moment the job really starts, so a cancel can
    # tell "the GPU work is running and will never be metered" (counted) from
    # "the request was still queued and nothing ran" (not counted).
    job_started = threading.Event()

    def _synth() -> SynthesisResult:
        job_started.set()
        return svc.synthesize(body.text, body.provider, body.voice, body.speed, body.ssml)

    try:
        # Admission lane `synth` (services/admission.py): bounded FIFO in front of
        # `_SYNTH_LOCK`, which stays INSIDE `svc.synthesize`. GpuBusy (503 +
        # Retry-After) and ClientGone (499, job skipped -- the disconnect check
        # runs at admission, before the job starts) propagate to the handlers
        # installed in main.py. The slot is held until the thread ends, even if
        # this coroutine is cancelled, so an abandoned job cannot let the next
        # one into a model that is still busy.
        result = await get_gpu_gate(request.app).run(SYNTH_LANE, _synth, request=request)
    except asyncio.CancelledError:
        # Caller went away. Count it ONLY when the job had started -- see
        # TTS_CANCELLED above for why this arm does nothing else -- and re-raise
        # unchanged. Listed first for legibility only: CancelledError derives
        # from BaseException (3.8+), so the `except Exception` families below
        # could never have caught it and this is not a behaviour change to them.
        if job_started.is_set():
            TTS_CANCELLED.inc()
        logger.info(
            "Synthesis cancelled by caller after %dms (%s) [provider=%s voice=%s text_len=%d]",
            int((time.monotonic() - started) * 1000),
            "running" if job_started.is_set() else "queued, skipped",
            body.provider,
            body.voice,
            len(body.text or ""),
        )
        raise
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GateClosedError as exc:
        # Shutting down: no new synthesis is admitted. It is a RuntimeError
        # subclass, so without this arm the generic one below turned a clean
        # shutdown into a 500 "Synthesis failed".
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, AssertionError) as exc:
        # Bad request: unknown voice / invalid input. AssertionError covers the
        # case where a provider lib (kokoro-onnx) asserts on an unknown voice
        # before our own registry validation can catch it.
        #
        # LOG IT. This branch was silent, and that made a whole failure class
        # invisible on the device. On 2026-08-25 the E2E `showcase` scenario died
        # on `tts /synthesize returned 400: need at least one array to
        # concatenate` — numpy's error for np.concatenate([]), raised inside the
        # VENDORED kokoro_onnx (Kokoro.create) when the phonemiser yields zero
        # batches for text that has no pronounceable content. Grepping mini2's
        # tts-backend logs for "concatenate" returned NOTHING: the ValueError went
        # straight out as a 400 with no record that it had ever happened. The
        # sibling RuntimeError branch below already logs; this one did not.
        #
        # warning, not exception: a 400 is usually a genuine client error and a
        # stack trace per bad request would be noise. The message is what matters,
        # because it is the only thing that distinguishes "unknown voice" from a
        # provider library failing on its own input.
        logger.warning(
            "Synthesis rejected (400): %s: %s [provider=%s voice=%s ssml=%s text_len=%d]",
            type(exc).__name__,
            exc,
            body.provider,
            body.voice,
            body.ssml,
            len(body.text or ""),
        )
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid synthesis request") from exc
    except RuntimeError as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    # Fire-and-forget cost event. cost_usd=0.0 (local synth); usage_extra
    # carries the audio metadata for downstream usage analytics. Failure
    # to emit must never block the synthesis response.
    metering_redis = getattr(request.app.state, "metering_redis", None)
    if metering_redis is not None:
        latency_ms = int((time.monotonic() - started) * 1000)
        # G5: declared per provider; an undeclared one RAISES rather than ship a
        # guessed funding billing would store verbatim.
        funding = funding_for_provider(result.provider)
        # G5: stable per request -- see `synthesis_event_key` for the precedence
        # (Idempotency-Key > X-Request-ID) and why it is never a body hash.
        event_key = synthesis_event_key(
            tenant_id=user.effective_tenant_id,
            idempotency_key=request.headers.get("idempotency-key"),
            request_id=getattr(request.state, "request_id", None),
        )
        # GC-safe fire-and-forget: schedule_cost_event holds a strong reference to
        # the background task so it can't be GC'd mid-flight (the old
        # asyncio.create_task(emit_cost_event(...)) handle was dropped → events
        # could be silently lost). Stream resolution + field marshalling live in
        # w2v_common.metering.
        schedule_cost_event(
            metering_redis,
            service_key=SERVICE_KEY,
            cost_source=COST_SOURCE,
            tenant_id=user.effective_tenant_id,
            # Canonical subject lives in the identity ContextVar, and NOTHING
            # else may stand in for it.
            #
            # This used to read `get_subject_id_from_context() or user.sub`.
            # On the bearer-licence lane -- the lane every production request
            # to this route takes -- `user.sub` is NOT a subject at all:
            # `AuthMiddleware` constructs `User(sub=str(tenant_id), ...)`, so
            # the fallback wrote the TENANT's uuid into billing's `subject_id`
            # column whenever the licence named no subject. That is a silent
            # substitution of one identity for another, and it is worse than a
            # gap: billing's unattributed counters read 100% attributed while
            # naming nobody, so the absence can never be found or repaired.
            # (kag shipped and fixed exactly this in c741bf9; the tenant uuids
            # it wrote are rows in `tenants` and absent from `subject`.)
            #
            # The ContextVar is already the vouched answer on every lane:
            # w2v_common deliberately empties it when introspection carried no
            # subject (`subject_id if subject_id != str(tenant_id) else
            # result.get("subject_id")`), and it is also where the delegation
            # lane's forwarded `X-Subject-Id` lands (middleware `_resolve_
            # license_identity`). So there is no truthful subject left for a
            # fallback to recover -- only untruthful ones.
            #
            # "" is the honest answer; billing stores it as NULL.
            subject_id=get_subject_id_from_context(),
            # The app the END USER is in, resolved from the licence
            # introspection this request already performed -- never asserted,
            # never defaulted, "" when it cannot be proven. See
            # `auth.dependencies.resolve_app_id` for why there is no header
            # rung and why this service's own app row must never be stamped
            # here. Omitting the kwarg is not equivalent to passing "": both
            # land NULL today, but a named "" is the statement that this route
            # considered the field, which is what stopped `license_id` above
            # from silently staying NULL for 1,120 of 1,120 events.
            app_id=resolve_app_id(request),
            # The licence comes from the SAME identity ContextVar the subject
            # above is read from -- w2v_common's AuthMiddleware populates both
            # from the introspection result. Omitting this kwarg did not fail or
            # warn: `w2v_common.metering.cost_bus` declares `license_id: str = ""`
            # with no context fallback, so every event this service has ever
            # emitted was written with a NULL licence.
            #
            # Measured on prod 2026-08-27: w2v-tts is 1,120 of 1,120 events over
            # 7 days with `license_id IS NULL` -- 100%. Its sibling `w2v-asr`,
            # running the IDENTICAL w2v_common AuthMiddleware, is 0% NULL over
            # 1,063 events. The only difference between them was this line
            # (`asr_backend/src/routers/asr_routes.py:120`).
            #
            # Licence-less spend is still billed and still capped, but
            # `CostDaily.license_attributed_clause` drops it from every
            # licence-keyed aggregate, so it can never reach an invoice line.
            license_id=get_license_id_from_context() or "",
            funding=funding,
            event_key=event_key,
            provider=result.provider,
            model=result.model or result.provider,
            input_tokens=len(body.text),
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            # billing's metadata allowlist (w2v-common 0.69.0, #168):
            # schema_id, job_id, fallback_used, stage, worker_name,
            # caller_client_id. Unknown keys are dropped at the emitter and a
            # None value is SKIPPED rather than stringified into the literal
            # "None" -- so absence is expressible, and that is the only reason
            # this dict is safe to build here.
            #
            # ONE of the six is knowable in this service. The other five are
            # deliberately NOT stamped, because a value invented to fill a
            # column reads 100% green on every coverage metric forever while
            # being wrong on every row -- strictly worse than the NULL it
            # replaces:
            #
            # * `fallback_used` -- this service HAS no fallback. `TTSService.
            #   synthesize` resolves exactly one provider and raises
            #   `ProviderUnavailableError` if it is not loaded; no alternative
            #   is ever attempted. A constant "False" would assert a fallback
            #   mechanism that does not exist, and would silently become a lie
            #   the day one is added.
            # * `job_id` / `schema_id` -- /synthesize is a synchronous call
            #   with neither. `X-Request-Id` is a correlation id, not a job id,
            #   and relabelling it as one would corrupt the grain for the
            #   services that do emit real job ids.
            # * `worker_name` -- a worker_management concept (the agent that
            #   ran the step). This process is a launchd service on mini2;
            #   stamping its hostname here would invent a second meaning for a
            #   column billing already reads with the first.
            # * `caller_client_id` -- the Keycloak `azp` would be the honest
            #   source, and it is not on `request.state`: `w2v_common`'s `User`
            #   does not carry it. Reachable only by widening w2v_common, which
            #   this change does not do.
            #
            # `stage` IS knowable and is already vouched by our own middleware:
            # `AuthMiddleware` calls `set_correlation_context(x-correlation-id,
            # x-stage)` for EVERY authenticated request before any lane runs,
            # so the ContextVar holds what the caller declared, and "" when
            # nothing was declared. `or None` turns that "" into a SKIPPED
            # field rather than an empty one.
            #
            # ⚠️ Expect this to read empty for now. The only caller of this
            # route is `modality_gateway`, whose `_FORWARD_REQUEST_HEADERS`
            # (src/services/tts_proxy.py) forwards x-tenant-id, x-license-id,
            # x-subject-id, x-effective-tenant-id, x-request-id, accept and
            # content-type -- no x-stage. The wiring is truthful and the
            # missing half is a gateway change, not one that belongs here.
            metadata_extra={"stage": get_stage_from_context() or None},
            usage_extra={
                "unit": "characters",
                "duration_s": result.duration_sec,
                "voice": result.voice,
                "speed": body.speed,
                "backend": result.provider,
            },
        )

    # A sample_rate that is not the engine's own is UNSERVICEABLE, not a formatting
    # detail. `encode_wav` -> `soundfile.write` does not resample: it stamps the same
    # samples with the requested rate, so a 48 kHz request against a 24 kHz engine
    # returned a clip that plays at DOUBLE speed and half the length — and billed it,
    # because `duration_sec` and the metered `usage_extra.duration_s` are both computed
    # from the sample COUNT at the engine rate. The caller got chipmunk audio, a 200,
    # and an invoice. Refuse it instead: this service has no resampler, and the honest
    # answer to "I cannot do that" is 400, not a corrupted file.
    if body.sample_rate is not None and body.sample_rate != result.sample_rate:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sample_rate={body.sample_rate} is not supported: this service does not "
                f"resample, and the {result.provider} engine produces {result.sample_rate} Hz. "
                f"Omit sample_rate, or request {result.sample_rate}."
            ),
        )
    target_sr = result.sample_rate
    # 32-bit FLOAT into the premaster, PCM_16 out of it — exactly ONE quantization,
    # and it is the LAST step. This used to encode PCM_16 here and normalize after,
    # so a quiet clip was truncated to 16 bits and THEN boosted by loudnorm, lifting
    # the quantization floor with the signal (grit/hiss in the pauses) — and pass-2's
    # `-c:a pcm_s16le` quantized a second, undithered time on top of that. The bytes
    # the caller receives are the same shape as before: `premaster_float_wav`
    # guarantees a PCM_16 WAV on every exit, fail-open included.
    float_wav = encode_wav(result.audio_pcm, target_sr, subtype="FLOAT")

    # Per-clip -18 LUFS pre-master (kill-switch TTS_PREMASTER_ENABLED, default on).
    # Fail-open: on any error premaster returns the (un-normalized) input audio with
    # empty metrics, so the response is never broken by the normalization step —
    # down-converted to PCM_16 first, so the caller's format never depends on it.
    # Offloaded to a thread: premaster runs two blocking ffmpeg passes, so calling
    # it directly would stall the event loop and serialize concurrent requests
    # It is CPU-only, so it runs on the default thread pool OUTSIDE the `synth`
    # admission lane: holding the GPU slot through two ffmpeg passes would idle the
    # model for no reason.
    wav_bytes, _m = await asyncio.to_thread(premaster_float_wav, float_wav, target_lufs=(body.master_lufs or -18.0))
    master_headers: dict[str, str] = {}
    if _m.integrated_lufs is not None:
        master_headers["X-TTS-LUFS"] = f"{_m.integrated_lufs:.2f}"
    if _m.true_peak_dbtp is not None:
        master_headers["X-TTS-TruePeak-dBTP"] = f"{_m.true_peak_dbtp:.2f}"
    # A fail-open clip is UN-NORMALISED audio returned with a 200. It used to be
    # visible to the caller only as the absence of X-TTS-LUFS — which is also what a
    # merely unparseable measurement looks like — and in the log only as a WARNING
    # latched once per process per reason, so clip #2 onward of a degraded process
    # was silent everywhere. Name it instead, on BOTH response branches (this dict
    # is spread into the raw-bytes headers and passed as `headers=` on the JSON
    # branch). Additive: a caller ignoring it is unaffected; a caller that wants to
    # refuse un-normalised narration can now see it.
    if _m.degraded_reason is not None:
        master_headers["X-TTS-Premaster-Degraded"] = _m.degraded_reason

    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        payload = SynthesizeResponse(
            audio_base64=base64.b64encode(wav_bytes).decode("ascii"),
            sample_rate=target_sr,
            duration_sec=result.duration_sec,
            provider=result.provider,
            voice=result.voice,
            format="wav",
            word_timings=_word_timings_in_output_timebase(result.word_timings, result.sample_rate, target_sr),
            speech_seconds=result.speech_seconds,
            pause_seconds=result.pause_seconds,
        )
        return Response(
            # exclude_none so `word_timings` is ABSENT (not `null`) on the ONNX
            # default path. Every other field on the model is required and
            # non-nullable, so nothing else can be dropped by this.
            content=payload.model_dump_json(exclude_none=True),
            media_type="application/json",
            headers=master_headers,
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-TTS-Provider": result.provider,
            "X-TTS-Voice": result.voice,
            "X-TTS-Sample-Rate": str(target_sr),
            "X-TTS-Duration-Sec": f"{result.duration_sec:.3f}",
            # Additive: a caller ignoring these headers is unaffected. See
            # SynthesizeResponse.speech_seconds/pause_seconds for the field semantics —
            # these are the same numbers, exposed on the raw-bytes path too.
            "X-TTS-Speech-Seconds": f"{result.speech_seconds:.3f}",
            "X-TTS-Pause-Seconds": f"{result.pause_seconds:.3f}",
            **master_headers,
        },
    )


@router.get(
    "/voices",
    response_model=VoicesResponse,
    operation_id="list_tts_voices",
    summary="List available voices + default provider/voice",
    responses={**AUTH_RESPONSES, 503: {"description": "Backend not initialised"}},
)
async def list_voices(
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> VoicesResponse:
    svc = _service_from_request(request)
    voices: list[VoiceInfo] = []
    for prov in svc.providers.values():
        ids = prov.voices or [prov.default_voice]
        for voice_id in ids:
            voices.append(VoiceInfo(voice_id=voice_id, provider=prov.name))
    return VoicesResponse(
        voices=voices,
        default_provider=svc.default_provider,
        default_voice=os.getenv("KOKORO_DEFAULT_VOICE", "af_heart"),
    )
