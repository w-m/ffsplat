"""Basic unit tests for a subset of helper functions in
``ffsplat.models.transformations``.
"""

import torch

from ffsplat.models.transformations import convert_to_dtype, minmax

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_minmax_scales_tensor_between_zero_and_one() -> None:
    """``minmax`` should rescale data to the interval ``[0, 1]``."""

    inp = torch.tensor([0.0, 1.0, 2.0])
    expected = torch.tensor([0.0, 0.5, 1.0])

    out = minmax(inp)

    assert torch.allclose(out, expected, atol=1e-6)


def test_convert_to_dtype_uint8() -> None:
    """``convert_to_dtype`` should cast to ``uint8`` when the data range is valid."""

    inp = torch.tensor([0, 127, 255])

    out = convert_to_dtype(inp, "uint8")

    assert out.dtype == torch.uint8
    assert torch.equal(out, inp.to(torch.uint8))
