"""SynthesizeRequest validation — bound sample_rate so a bad value 4xx's, not 500s."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import SynthesizeRequest


def test_sample_rate_defaults_to_none() -> None:
    assert SynthesizeRequest(text="hello").sample_rate is None


def test_sample_rate_accepts_valid_audio_rate() -> None:
    assert SynthesizeRequest(text="hello", sample_rate=24000).sample_rate == 24000


@pytest.mark.parametrize("bad", [-1, 0, 100, 96000, 2**31])
def test_sample_rate_rejects_out_of_range(bad: int) -> None:
    # An unbounded sample_rate flowed into soundfile.write(sr=...) OUTSIDE the
    # route's try/except → a negative value 500'd a request that should be 4xx, and
    # a huge value produced a malformed WAV with a 200. Bound it at the schema.
    with pytest.raises(ValidationError):
        SynthesizeRequest(text="hello", sample_rate=bad)


def test_mp3_is_no_longer_advertised_by_the_schema() -> None:
    """The schema promised an encoder that does not exist.

    `output_format: "mp3"` was accepted and then ignored: the response carried WAV
    bytes, media_type audio/wav and format "wav" — a 200 contradicting the request,
    which the caller had no way to detect. Narrowing the Literal makes pydantic
    answer with the supported value instead of the service lying.
    """
    import pytest
    from pydantic import ValidationError

    from models import SynthesizeRequest

    # Validate a JSON body, which is what the HTTP layer actually does — and what
    # lets this assert the rejection at all, since the narrowed Literal makes the
    # keyword form a static type error rather than a runtime one.
    SynthesizeRequest.model_validate({"text": "Hello.", "output_format": "wav"})
    with pytest.raises(ValidationError):
        SynthesizeRequest.model_validate({"text": "Hello.", "output_format": "mp3"})
