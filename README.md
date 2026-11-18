# tts-backend

## Native ONNX word timings and custom voices

Both gateway Macs use `TTS_ENGINE=kokoro-onnx` with the same duration-bearing
graph. `kokoro-onnx>=0.6.1` maps phoneme durations to misaki word tokens; the
service preserves timing offsets through SSML breaks, edge trimming, content
pauses and crossfades. No Whisper or PyTorch synthesis pass is needed.

The pinned graph is the `model-files-v1.1` release's `kokoro-v1.0.onnx`:
SHA256 `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a`.
Deploy it under `kokoro-v1.0-timed-beb0d1848dee.onnx`, because older audio-only
graphs use the same upstream filename. Set `KOKORO_REQUIRE_TIMINGS=true` so a
wrong graph cannot silently remove caption support. `/health` now reports
`provider_engines` and `word_timings_available` from the loaded model.

Set `KOKORO_CUSTOM_VOICES_DIR` to a persistent directory outside the checkout.
Place versioned `.npy` profiles there (e.g. `am_george_trained_v3.npy`) and restart
the service. Each must be a finite float32 `[510,1,256]` tensor, loaded without
pickle. IDs are validated; stock IDs cannot be overwritten. Registered profiles
are listed by `/api/tts/voices`; synthesis keeps the public ID in its response.
The local generator's portable `.npy` output needs no conversion.

Use a fresh voice ID for each revision because downstream audio caches key on
the voice string. W2V's project/casting catalog also needs to authorize a custom
ID before its UI can select it. Raw reference recordings stay on the generator.

Before a rollout run `uv run pytest`, `ruff check`, `ty check` and the live-model
smoke script `PYTHONPATH=src python scripts/smoke-timed-onnx.py` with model/voice
paths in the environment. Roll the Macs one at a time and verify authenticated
JSON synthesis returns nonempty, bounded word timings as well as WAV audio.

Text-to-speech inference service for Apple Silicon. Replaces an earlier in-cluster TTS deployment.

| | |
|---|---|
| Engines | Kokoro (`kokoro-onnx`), Kitten (`kittentts`) |
| Default provider | `kokoro` |
| Port | `9202` |
| Hostname | `tts.test` |
| Auth | Istio header pass-through |

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/tts/synthesize` | JSON in (`text`, optional `provider`/`voice`/`speed`/`sample_rate`/`output_format`); WAV bytes out (or `application/json` with base64 if Accept header asks for it) |
| `GET` | `/api/tts/voices` | available voices per loaded provider |
| `GET` | `/api/tts/health` | provider-load status |
| `GET` | `/health{,/ready,/live}` | service-level health |

Response headers on the audio stream: `X-TTS-Provider`, `X-TTS-Voice`, `X-TTS-Sample-Rate`, `X-TTS-Duration-Sec`.

## Configuration

| Var | Default | Purpose |
|---|---|---|
| `TTS_DEFAULT_PROVIDER` | `kokoro` | which provider to use when caller doesn't specify |
| `KOKORO_ENABLED` | `true` | load Kokoro at startup |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | fallback voice |
| `KOKORO_MODEL_PATH` | `~/Library/Caches/tts/kokoro-v1.0.onnx` | model weights |
| `KOKORO_VOICES_PATH` | `~/Library/Caches/tts/voices-v1.0.bin` | voice embeddings |
| `KITTEN_ENABLED` | `true` | load Kitten at startup |
| `KITTEN_MODEL_ID` | `KittenML/kitten-tts-mini-0.8` | HF repo |
| `TTS_SAMPLE_RATE` | `24000` | default output sample rate |
| `LOG_LEVEL` | `INFO` | |
| `CORS_ORIGINS` | `https://api.nonagentic.ai,https://w2v.nonagentic.ai` | |
| `METERING_REDIS_URL` | (unset) | `redis://default:<pw>@metering-redis.test:31279/0` — when set, emits a cost event to `metering:costs` per successful synthesis (`cost_source=w2v-tts`, `cost_usd=0.0` since both providers run locally). Unset = no-op. |

Production values land in `~/.config/tts-backend/env` (with `/etc/tts-backend/env` as a fallback for privileged-provisioning hosts). `start.sh` reads whichever path it finds.

## Local dev

```bash
uv sync
PYTHONPATH=src uv run pytest                                      # CI-safe
PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 9202   
```

First run needs the Kokoro ONNX model + voices file downloaded to `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH`. See `scripts/fetch-models.sh` (TODO; see Phase 4 of the install plan).

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.
