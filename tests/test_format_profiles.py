"""End-to-end tests for the output-format templates in ``ffsplat/conf/format``.

The templates are YAML scene graphs, so a shape/field-name typo in one of them is only
detected when that profile is actually run. These tests build a small synthetic 3DGS scene
with the same field layout the 3DGS-INRIA PLY decoder produces (``sh`` of shape ``[N, 16, 3]``,
i.e. degree 3) and push it through every profile whose codebook does not depend on the full
sized scene, then decode the container back and check the round-trip invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffsplat.coding.scene_decoder import decode_gaussians
from ffsplat.coding.scene_encoder import encode_gaussians
from ffsplat.models.fields import Field
from ffsplat.models.gaussians import Gaussians
from ffsplat.models.operations import Operation

# 1000 primitives: not a perfect square, so the SOG-* profiles exercise the
# `prune_by: opacities` perfect-square pruning step as well.
N = 1000


def _synthetic_scene(n: int = N) -> Gaussians:
    """A float32 3DGS scene with the same shapes the INRIA PLY decoder produces."""
    generator = torch.Generator().manual_seed(0)
    op = Operation(input_fields={}, params={"from": {"file": "synthetic"}})
    means = 2.0 * torch.rand(n, 3, generator=generator) - 1.0
    quaternions = torch.randn(n, 4, generator=generator)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    scales = torch.rand(n, 3, generator=generator) * 0.1
    opacities = torch.rand(n, 1, generator=generator)
    sh = torch.randn(n, 16, 3, generator=generator)
    return Gaussians(
        means=Field(means, op),
        quaternions=Field(quaternions, op),
        scales=Field(scales, op),
        opacities=Field(opacities, op),
        sh=Field(sh, op),
    )


def _ply_properties(path: Path) -> list[str]:
    """The vertex property names of a binary little-endian PLY, in file order."""
    with path.open("rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
    return [line.split()[-1] for line in header.decode().splitlines() if line.startswith("property float ")]


def _roundtrip(tmp_path: Path, output_format: str) -> Gaussians:
    """Encode with ``output_format``, decode the container back, return the decoded scene."""
    out = tmp_path / output_format
    encode_gaussians(_synthetic_scene(), out, output_format, False)
    return decode_gaussians(out, "smurfx", verbose=False)


def test_sog_web_encodes_and_decodes(tmp_path: Path) -> None:
    """Baseline: the reference SOG profile."""
    decoded = _roundtrip(tmp_path, "SOG-web")
    assert decoded.num_gaussians == 961  # 1000 pruned to the largest perfect square
    assert decoded.sh.data.shape[1] == 16
    assert decoded.means.data.dtype == torch.float32


def test_sog_web_nosh_encodes_and_decodes(tmp_path: Path) -> None:
    """The no-SH profile keeps only f_dc; the `sh` split must still cover all 16 coefficients."""
    out = tmp_path / "sog"
    encode_gaussians(_synthetic_scene(), out, "SOG-web-nosh", False)
    assert sorted(p.name for p in out.iterdir()) == [
        "container_meta.yaml",
        "f_dc.avif",
        "means_bytes_0.avif",
        "means_bytes_1.avif",
        "opacities.avif",
        "quaternions.avif",
        "scales.avif",
    ]
    decoded = decode_gaussians(out, "smurfx", verbose=False)
    assert decoded.num_gaussians == 961
    assert decoded.sh.data.shape == (961, 1, 3)


def test_3dgs_inria_nosh_ply_writes_only_the_dc_coefficients(tmp_path: Path) -> None:
    """The no-SH PLY profile drops f_rest; the same [1] vs [1, 15] split bug hit this file."""
    out = tmp_path / "ply_nosh"
    encode_gaussians(_synthetic_scene(), out, "3DGS-INRIA-nosh-ply", False)
    props = _ply_properties(out / "point_cloud.ply")
    assert sorted(props) == sorted([
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "scale_0",
        "scale_1",
        "scale_2",
        "opacity",
    ])
    # and it must decode again through the matching 3DGS-INRIA-nosh decoding template
    # the PLY profiles do not sort/prune, so the primitive count is unchanged
    decoded = decode_gaussians(out / "point_cloud.ply", "3DGS-INRIA-nosh.ply", verbose=False)
    assert decoded.num_gaussians == N
    assert decoded.sh.data.shape == (N, 1, 3)


@pytest.mark.parametrize(
    ("profile", "expected_files"),
    [
        ("SOG-web-png", ["f_rest_centroids.png"]),
        ("SOG-web-sh-split", ["f_rest_0.avif", "f_rest_14.avif"]),
    ],
)
def test_sog_web_variants_encode_and_decode(tmp_path: Path, profile: str, expected_files: list[str]) -> None:
    """The png/avif-SH variants share the SOG-web codec but write different image sets."""
    out = tmp_path / profile
    encode_gaussians(_synthetic_scene(), out, profile, False)
    names = {p.name for p in out.iterdir()}
    assert "container_meta.yaml" in names
    assert set(expected_files) <= names, f"{profile} is missing {set(expected_files) - names}"
    decoded = decode_gaussians(out, "smurfx", verbose=False)
    assert decoded.num_gaussians == 961
    assert decoded.sh.data.shape[1] == 16
    assert decoded.means.data.dtype == torch.float32


def test_sog_playcanvas_requires_the_full_codebook(tmp_path: Path) -> None:
    """SOG-PlayCanvas packs a fixed 1024x960 codebook texture (65536 clusters of 45 SH).

    Scenes with fewer primitives get a smaller codebook from the k-means clamp, so the profile
    cannot be used on them. The reshape must say so instead of raising a bare RuntimeError.
    """
    with pytest.raises(ValueError, match="Cannot reshape field 'shN_centroids'"):
        encode_gaussians(_synthetic_scene(), tmp_path / "playcanvas", "SOG-PlayCanvas", False)
