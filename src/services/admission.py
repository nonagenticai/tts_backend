"""GPU admission control for synthesis (w2v_common.admission, contract 2026-10-15).

ONE lane, ``synth``, guards every call into a TTS model. It bounds what the
process-wide ``_SYNTH_LOCK`` (services/tts_service.py) never could: the lock
serialises inference -- and must stay, concurrent kokoro/espeak garbles audio --
but its waiters were unbounded, had no deadline, and a request whose client had
already hung up still synthesised. The lane puts a bounded FIFO in front of the
lock: at most ``max_active`` running, ``max_waiting`` queued, the rest an
immediate 503 + ``Retry-After``; a request that waited past ``max_wait_s`` gets
the same 503, and a request whose client is gone when its turn comes is skipped.

Code defaults are the issue's numbers (1 / 16 / 20 s). Production overrides come
from the host env: ``GPU_LANE_SYNTH_MAX_ACTIVE|_MAX_WAITING|_MAX_WAIT_S|_SEED_S``.
``max_active`` above 1 buys nothing while the lock is held inside the job -- the
extra executor threads would only queue on the lock, invisibly to the gate.

The gate lives on ``app.state`` (built in the lifespan, so a malformed env value
fails the boot loudly), not at module import: its semaphore binds to the event
loop that first contends for it.
"""

from __future__ import annotations

from typing import Any

from w2v_common.admission import AdmissionGate, LaneConfig

SYNTH_LANE = "synth"
SYNTH_LANE_DEFAULT = LaneConfig(max_active=1, max_waiting=16, max_wait_s=20.0)


def build_gpu_gate() -> AdmissionGate:
    """A fresh gate with the code defaults, env overrides applied."""
    return AdmissionGate({SYNTH_LANE: SYNTH_LANE_DEFAULT})


def get_gpu_gate(app: Any) -> AdmissionGate:
    """The app's gate, created on first use when the lifespan did not run (tests)."""
    gate = getattr(app.state, "gpu_gate", None)
    if gate is None:
        gate = build_gpu_gate()
        app.state.gpu_gate = gate
    return gate
