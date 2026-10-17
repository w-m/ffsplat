"""Tests for :mod:`ffsplat.io.ply`.

Regression coverage for :func:`encode_ply` accepting the trailing singleton dimension
that decode paths leave on some fields (``scene_decoder._create_scene`` intentionally
unsqueezes ``opacities`` to ``[N, 1]``), while still rejecting genuine multi-component
fields instead of letting numpy raise an opaque broadcast error.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ffsplat.io.ply import decode_ply, encode_ply
from ffsplat.models.fields import Field

OP = SimpleNamespace()


def test_encode_ply_accepts_scalar_and_trailing_singleton_fields(tmp_path) -> None:
    path = tmp_path / "point_cloud.ply"
    fields = {
        "x": Field(torch.arange(4, dtype=torch.float32), OP),
        "opacity": Field(torch.arange(4, dtype=torch.float32).reshape(4, 1), OP),
    }

    encode_ply(fields, path)

    decoded = decode_ply(path, "")
    assert set(decoded) == {"x", "opacity"}
    for name in fields:
        assert decoded[name].data.shape == (4,)
        assert decoded[name].data.dtype == torch.float32
        assert torch.equal(decoded[name].data, fields[name].data.reshape(4))


def test_encode_ply_rejects_multi_component_fields(tmp_path) -> None:
    fields = {"x": Field(torch.zeros(4, 3), OP)}
    with pytest.raises(ValueError, match="must be scalar"):
        encode_ply(fields, tmp_path / "bad.ply")
