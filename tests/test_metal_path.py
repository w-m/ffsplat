"""Guard: ffsplat's rasterisation calls must run on mpsplat's native Metal kernels.

`mpsplat` routes a call onto its pure-torch MPS rasteriser *silently* whenever an argument
falls outside what the Metal kernels support (its `gsplat/cuda/_wrapper.py`):

* projection / intersection (`_wrapper.py:620-633`, `:850-865`): ``covars is None``, not ``packed``,
  ``calc_compensations`` False, ``camera_model == "pinhole"``, ``len(batch_dims) <= 1``.
  ``calc_compensations`` is True exactly for ``rasterize_mode="antialiased"``.
* rasterisation (`_wrapper.py:1102-1131`): ``tile_size == 16``, ``channels == 3``, not ``packed``,
  ``backgrounds is None``, ``masks is None``.

The fallbacks are several times slower on MPS, so this test traps them and asserts that ffsplat's
three real render entry points instead reach the Metal kernels. It runs only where MPS exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ffsplat.cli.eval import rasterize_splats
from ffsplat.cli.view import render_fn as view_render_fn
from ffsplat.models.fields import Field
from ffsplat.models.gaussians import Gaussians
from ffsplat.models.operations import Operation
from ffsplat.render.viewer import CameraState

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required for the Metal path")

N = 512
IMG_WH = (96, 64)


def _scene() -> Gaussians:
    """A small real-layout 3DGS scene (sh degree 3) on MPS."""
    generator = torch.Generator().manual_seed(0)
    op = Operation(input_fields={}, params={"from": {"file": "synthetic"}})
    means = 0.5 * (2.0 * torch.rand(N, 3, generator=generator) - 1.0)
    quats = torch.randn(N, 4, generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    # big enough to survive cli/view.py's radius_clip=3 at this camera distance
    scales = 0.05 + 0.15 * torch.rand(N, 3, generator=generator)
    opacities = torch.rand(N, 1, generator=generator)
    sh = torch.randn(N, 16, 3, generator=generator)
    scene = Gaussians(
        means=Field(means, op),
        quaternions=Field(quats, op),
        scales=Field(scales, op),
        opacities=Field(opacities, op),
        sh=Field(sh, op),
    )
    return scene.to("mps")


def _camera_state() -> CameraState:
    """A camera 2 units in front of the scene (3DGS convention: camera looks along +z)."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [0.0, 0.0, -2.0]
    return CameraState(fov=1.0, aspect=IMG_WH[0] / IMG_WH[1], c2w=c2w)


def _install_probe(monkeypatch) -> dict[str, list[str]]:
    """Trap the three torch fallbacks and count the Metal entry points."""
    from gsplat.cuda import _torch_impl, _torch_rasterize
    from gsplat.mps import _intersect, _projection, _rasterize

    hits: dict[str, list] = {"fallback": [], "metal": [], "kwargs": []}

    def make_trap(name):
        def trap(*_args, **_kwargs):
            hits["fallback"].append(name)
            raise AssertionError(f"rasterisation fell back to {name} instead of the Metal kernel")

        return trap

    for module, name in (
        (_torch_impl, "_fully_fused_projection"),
        (_torch_impl, "_isect_tiles"),
        (_torch_rasterize, "_rasterize_to_pixels_torch"),
    ):
        monkeypatch.setattr(module, name, make_trap(name))

    def make_counter(name):
        original = getattr(module_of(name), name)

        def counted(*args, **kwargs):
            hits["metal"].append(name)
            return original(*args, **kwargs)

        return counted

    def module_of(name: str):
        return {
            "rasterize_to_pixels_3dgs_metal": _rasterize,
            "fully_fused_projection_3dgs_metal": _projection,
            "_intersect_tile_metal": _intersect,
        }[name]

    for name in ("rasterize_to_pixels_3dgs_metal", "fully_fused_projection_3dgs_metal", "_intersect_tile_metal"):
        monkeypatch.setattr(module_of(name), name, make_counter(name))

    # record the high-level kwargs ffsplat passes, so the *argument policy* that decides which
    # path is taken is pinned too (tile_size must stay the 16 default, rasterize_mode "classic",
    # no backgrounds/masks, sh_degree <= 4 -- mps/_sh.py:28 asserts K <= 25).
    import ffsplat.cli.eval as eval_mod
    import ffsplat.cli.live as live_mod
    import ffsplat.cli.view as view_mod

    for module in (view_mod, live_mod, eval_mod):
        original = module.rasterization

        def recorded(*args, _original=original, **kwargs):
            hits["kwargs"].append(kwargs)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "rasterization", recorded)
    return hits


def _check_argument_policy(hits: dict) -> None:
    """The arguments ffsplat hands to rasterization() must stay inside the Metal kernels' envelope."""
    assert hits["kwargs"], "rasterization was never called"
    for kw in hits["kwargs"]:
        assert kw.get("backgrounds") is None, "backgrounds= drops the call onto the torch rasteriser"
        assert kw.get("masks") is None, "masks= drops the call onto the torch rasteriser"
        assert "tile_size" not in kw, "tile_size must stay the default (only 16 keeps the Metal path)"
        assert "rasterize_mode" not in kw, "antialiased would set calc_compensations and leave Metal"
        assert "packed" not in kw, "packed=True raises NotImplementedError on the MPS port"
        assert kw.get("sh_degree") is None or kw["sh_degree"] <= 4, "mps/_sh.py:28 allows degree <= 4"


def test_view_render_fn_uses_the_metal_kernels(monkeypatch) -> None:
    hits = _install_probe(monkeypatch)
    scene = _scene()
    out = view_render_fn(
        scene.means.data,
        scene.quaternions.data,
        scene.scales.data,
        scene.opacities.data,
        scene.sh.data,
        scene.sh_degree,
        _camera_state(),
        IMG_WH,
    )
    assert hits["fallback"] == []
    assert float(out.mean()) > 0.0, "the render is empty"
    assert "rasterize_to_pixels_3dgs_metal" in hits["metal"]
    assert "fully_fused_projection_3dgs_metal" in hits["metal"]
    _check_argument_policy(hits)


def test_eval_rasterize_splats_stays_on_the_metal_path_with_a_white_background(monkeypatch) -> None:
    """`eval --white-background` used to pass `backgrounds=`, which drops off the Metal path."""
    hits = _install_probe(monkeypatch)
    scene = _scene()
    camtoworlds = torch.eye(4, device="mps").repeat(1, 1, 1)
    camtoworlds[0, 2, 3] = -2.0
    K = torch.tensor([[58.0, 0.0, IMG_WH[0] / 2], [0.0, 58.0, IMG_WH[1] / 2], [0.0, 0.0, 1.0]], device="mps")[None]
    colors, alphas, _ = rasterize_splats(
        gaussians=scene,
        camtoworlds=camtoworlds,
        Ks=K,
        width=IMG_WH[0],
        height=IMG_WH[1],
        use_white_background=True,
    )
    assert hits["fallback"] == []
    assert "rasterize_to_pixels_3dgs_metal" in hits["metal"]
    assert float(alphas.max()) > 0.0, "the render is empty"
    # pixels nothing was rendered into must be pure white: that only happens if the background was
    # composited in afterwards (with backgrounds=None the rasterizer returns the raw accumulated
    # colour, i.e. 0 there).
    empty = alphas[..., 0] < 1e-4
    assert bool(empty.any()), "no empty pixels to check the background on"
    assert float((colors[..., 0][empty] - 1.0).abs().max()) < 1e-3, "white background not composited"
    _check_argument_policy(hits)


def test_live_render_fn_uses_the_metal_kernels(monkeypatch) -> None:
    from ffsplat.cli.live import InteractiveConversionTool

    hits = _install_probe(monkeypatch)
    scene = _scene()
    out = InteractiveConversionTool.render_fn(
        object(),  # the method never touches self
        scene.means.data,
        scene.quaternions.data,
        scene.scales.data,
        scene.opacities.data,
        scene.sh.data,
        scene.sh_degree,
        _camera_state(),
        IMG_WH,
    )
    assert hits["fallback"] == []
    assert float(out.mean()) > 0.0
    assert "rasterize_to_pixels_3dgs_metal" in hits["metal"]
    _check_argument_policy(hits)


def test_sh_degree_above_four_is_announced(monkeypatch) -> None:
    """The one silent fallback left in ffsplat's call chain: SH degree > 4.

    `mpsplat`'s `gsplat/cuda/_wrapper.py:374-387` only routes `spherical_harmonics` to the Metal
    kernel while `coeffs.shape[-2] <= 25`; beyond that it evaluates SH with the pure-torch
    implementation without any error. `ffsplat.device.warn_if_not_metal` now announces it, and the
    projection/rasterisation kernels stay on Metal.
    """
    from gsplat.cuda import _torch_impl
    from gsplat.cuda._wrapper import spherical_harmonics

    hits = _install_probe(monkeypatch)
    torch_fallback_sh = []
    original = _torch_impl._spherical_harmonics
    monkeypatch.setattr(
        _torch_impl,
        "_spherical_harmonics",
        lambda *a, **k: (torch_fallback_sh.append(1), original(*a, **k))[1],
    )

    generator = torch.Generator().manual_seed(0)
    scene = _scene()
    degree5 = torch.randn(N, 36, 3, generator=generator).to("mps")  # degree 5 -> K=36 > 25
    assert spherical_harmonics(5, torch.ones(1, 3, device="mps"), degree5[:1]).shape == (1, 3)

    with pytest.warns(RuntimeWarning, match="sh_degree=5"):
        out = view_render_fn(
            scene.means.data,
            scene.quaternions.data,
            scene.scales.data,
            scene.opacities.data,
            degree5,
            5,
            _camera_state(),
            IMG_WH,
        )
    assert float(out.mean()) > 0.0
    assert torch_fallback_sh, "degree 5 did not reach the torch SH path (probe assumption wrong)"
    assert "rasterize_to_pixels_3dgs_metal" in hits["metal"], "the rasteriser left Metal"
    assert hits["fallback"] == []
