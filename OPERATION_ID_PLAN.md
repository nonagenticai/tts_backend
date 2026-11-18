# Operation ID Plan — tts_backend

Source of truth for FastAPI `operation_id=` values. Mirrored in
`tests/test_openapi.py::EXPECTED_OPERATION_IDS`. Globally unique across the
mini1/mini2 host-services fleet.

Naming convention: `verb_noun` snake_case.

## `src/main.py`

| Method | Path           | Function | operation_id              |
| ------ | -------------- | -------- | ------------------------- |
| GET    | /              | root     | service_root              |
| GET    | /health        | health   | health_check              |
| GET    | /health/ready  | ready    | health_ready              |
| GET    | /health/live   | live     | health_live               |
| GET    | /me            | me       | get_current_identity      |

## `src/routers/tts_routes.py` (prefix `/api/tts`, tag `TTS`)

| Method | Path                  | Function    | operation_id      |
| ------ | --------------------- | ----------- | ----------------- |
| POST   | /api/tts/synthesize   | synthesize  | tts_synthesize    |
| GET    | /api/tts/voices       | list_voices | list_tts_voices   |

Total: 7 explicit `operation_id` values.

The previously-existing GET `/api/tts/health` (which shadowed the canonical
`/health` route) was dropped — keep all health-probe traffic on the main
`/health*` endpoints registered in `main.py`.
