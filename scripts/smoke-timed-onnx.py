"""Live-model smoke, separate from unit tests which stub kokoro_onnx.

Set KOKORO_MODEL_PATH, KOKORO_VOICES_PATH; PYTHONPATH=src python scripts/smoke-timed-onnx.py.
"""

import json
import os

import numpy as np

from services.tts_service import KokoroProvider

os.environ["TTS_ENGINE"] = "kokoro-onnx"
os.environ["KOKORO_REQUIRE_TIMINGS"] = "true"
provider = KokoroProvider()
provider.load()
cases = [
    ("Hello world. This is a word timing check.", 1.0),
    ('<speak>One two.<break time="500ms"/>Three <emphasis>four</emphasis> five.</speak>', 1.0),
    ("The meeting starts at 10:30. The total is 42 dollars.", 1.3),
    ("Go go go! Don't repeat a word by accident.", 0.8),
]
out = []
for text, speed in cases:
    result = provider.synthesize(text, "af_heart", speed)
    assert result.word_timings, text
    assert np.isfinite(result.audio_pcm).all()
    assert all(0 <= w.start <= w.end <= result.duration_sec for w in result.word_timings)
    assert all(a.start <= b.start for a, b in zip(result.word_timings, result.word_timings[1:], strict=False))
    out.append(
        {
            "text": text,
            "speed": speed,
            "seconds": result.duration_sec,
            "words": [{"word": w.word, "start": w.start, "end": w.end} for w in result.word_timings],
        }
    )
print(json.dumps({"engine": provider.engine, "has_timings": provider.emits_word_timings(), "cases": out}, indent=2))
