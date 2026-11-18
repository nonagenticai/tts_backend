#!/usr/bin/env python3
"""Real-model smoke test: POST /api/tts/synthesize through the app's own code path.

Unit tests stub ``kokoro_onnx`` (tests/conftest.py), so a dependency upgrade
(onnxruntime, kokoro-onnx, huggingface-hub, misaki/spacy, fastapi/starlette,
numpy/soundfile) is never exercised against a real model there. This script is
the repeatable check to run on a dev Mac (NOT on mini1/mini2) before merging a
lock change, because a merge auto-restarts tts-backend on both hosts.

It runs the REAL app in-process: lifespan (model load) -> route -> admission
gate -> TTSService -> kokoro-onnx -> premaster/ffmpeg, with only
``get_current_user`` overridden to a fixed test tenant (the route needs an
authenticated principal; ENABLE_AUTH=false skips the auth middleware, as in
tests/conftest.py; nothing else is mocked). No port is opened.

Usage (model files are NOT in the repo, ~340 MB, download once):
    curl -L -o /tmp/tts/kokoro-v1.0.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    curl -L -o /tmp/tts/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
    KOKORO_MODEL_PATH=/tmp/tts/kokoro-v1.0.onnx KOKORO_VOICES_PATH=/tmp/tts/voices-v1.0.bin \
      TTS_G2P=misaki uv run python scripts/smoke_synthesize.py

Exits non-zero unless every request returns 200 with a decodable, non-silent
WAV whose duration is plausible for the text. Prints one JSON line with the
measurements (duration, wall time, peak RSS) so two locks can be compared.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

os.environ.setdefault("ENVIRONMENT", "development")
# Skips the auth MIDDLEWARE only (grants nothing); the route principal comes
# from the get_current_user override below, exactly as tests/conftest.py does.
os.environ.setdefault("ENABLE_AUTH", "false")
os.environ.setdefault("TTS_ENGINE", "kokoro-onnx")
os.environ.setdefault("KOKORO_ENABLED", "true")
os.environ.setdefault("KITTEN_ENABLED", "false")
os.environ.setdefault("W2V_LICENSE_GATE_ENABLED", "false")
for _brew in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _brew not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + _brew

TEXTS = [
    "Hello world. This is a short smoke test of the speech backend.",
    "On March 3rd, 2026, the C-F-O met 12 engineers at 9:30 a.m. to review a $4.5 million budget.",
]


def _peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


async def main() -> int:
    import httpx
    import numpy as np
    import soundfile as sf

    from auth.dependencies import get_current_user
    from auth.models import User
    from main import app, lifespan

    tenant = "11111111-1111-4111-8111-111111111111"
    app.dependency_overrides[get_current_user] = lambda: User(
        sub=tenant, preferred_username="smoke", tenant_id=tenant, roles=["admin"]
    )
    rss_before = _peak_rss_mib()
    t_load = time.monotonic()
    results: list[dict] = []
    ok = True
    async with lifespan(app):
        load_s = time.monotonic() - t_load
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://smoke", timeout=300) as ac:
            for text in TEXTS:
                t0 = time.monotonic()
                resp = await ac.post("/api/tts/synthesize", json={"text": text})
                wall = time.monotonic() - t0
                row: dict = {"status": resp.status_code, "wall_s": round(wall, 2), "chars": len(text)}
                if resp.status_code != 200:
                    row["error"] = resp.text[:300]
                    ok = False
                else:
                    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
                    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
                    dur = len(mono) / sr
                    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
                    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
                    # ~10-25 chars/s of English speech; generous bounds.
                    plausible = len(text) / 30 <= dur <= len(text) / 5
                    row.update(
                        sr=sr,
                        duration_s=round(dur, 2),
                        peak=round(peak, 3),
                        rms=round(rms, 4),
                        lufs=resp.headers.get("X-TTS-LUFS"),
                        plausible=plausible,
                    )
                    if not (plausible and peak > 0.05 and rms > 0.005 and np.isfinite(mono).all()):
                        ok = False
                results.append(row)
    print(
        json.dumps(
            {
                "ok": ok,
                "model_load_s": round(load_s, 2),
                "peak_rss_mib_before_load": round(rss_before, 1),
                "peak_rss_mib": round(_peak_rss_mib(), 1),
                "requests": results,
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
