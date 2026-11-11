"""A request keys an image's placeholders in its prefix by the image's content and grid."""

from dataclasses import replace

import torch

from cti.inference._control._scheduling.request import image_key_ids
from cti.multimodal import ImageSpan

_DIGEST = bytes(range(32))


def test_an_image_span_keys_its_tokens_by_content_and_grid_alone() -> None:
    span = ImageSpan(3, 20, (1, 8, 10), _DIGEST)

    key = image_key_ids(span)

    # Negative int32 words, so no text token equals one, repeated over the span.
    assert key.dtype == torch.int32 and len(key) == 20 and bool((key < 0).all())
    assert torch.equal(key[8:16], key[:8]) and torch.equal(key[16:], key[:4])
    assert torch.equal(image_key_ids(replace(span, offset=40)), key)
    for other in (replace(span, digest=bytes(32)), replace(span, grid_thw=(1, 10, 8))):
        assert image_key_ids(other)[0] != key[0]
