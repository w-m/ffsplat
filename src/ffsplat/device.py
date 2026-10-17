"""Backend-capability helpers.

PyTorch's Apple-Silicon (MPS) backend can represent fewer tensor dtypes than the
CPU and CUDA backends. Notably ``uint16`` and ``uint32`` are missing::

    TypeError: Trying to convert UInt16 to the MPS backend but it does not have
    support for that dtype.

The SOG encoders use those widths as *intermediate carriers* while a value is
split into bytes (``SimpleQuantize(dtype="uint16") -> SplitBytes``), and the
decoder rebuilds the value from the byte images with ``Combine(method="bytes")``
into ``int32``/``int64``. Nothing therefore depends on the tensor actually
carrying an ``uint16`` dtype, only on the byte pattern.
"""

from __future__ import annotations

import warnings

import torch

#: dtypes the MPS backend cannot represent, mapped to an exact substitute.
#: The substitutes have a superset of the source range and the same two's
#: complement byte layout for the non-negative values the callers range-check for.
_MPS_DTYPE_SUBSTITUTES: dict[torch.dtype, torch.dtype] = {
    torch.uint16: torch.int32,
    torch.uint32: torch.int64,
}


def encoding_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    """Return a dtype equivalent to ``dtype`` that ``device`` can represent.

    On CPU/CUDA this is the identity. On MPS, dtypes the backend lacks are
    replaced by a wider signed type; the substitution is exact for the
    byte-splitting codecs (see the module docstring).
    """
    if device.type == "mps":
        return _MPS_DTYPE_SUBSTITUTES.get(dtype, dtype)
    return dtype


def _best_available() -> torch.device:
    """The best accelerator this machine actually has: MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve the device ffsplat should run on.

    * nothing requested -> MPS if available, else CUDA if available, else CPU;
    * something requested that this machine can actually use -> that device;
    * something requested that it cannot (the historical hard-coded ``"cuda"`` /
      ``"cuda:0"`` on an Apple-Silicon Mac) -> warn once and fall back to the best
      available device, so one and the same command line keeps working everywhere.

    ``cuda`` is therefore only ever returned when ``torch.cuda.is_available()`` is true,
    and ``mps`` only when the MPS backend is built and available; Linux/CUDA behaviour is
    unchanged by construction.
    """
    if requested is None:
        return _best_available()

    device = torch.device(requested)
    usable = (
        (device.type == "cuda" and torch.cuda.is_available())
        or (device.type == "mps" and torch.backends.mps.is_available())
        or device.type in ("cpu", "meta")
    )
    if usable:
        return device

    fallback = _best_available()
    print(
        f"[device] '{device}' is not usable on this machine; using '{fallback}' instead.",
        flush=True,
    )
    return fallback


def synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has finished (no-op on CPU).

    Routing through here keeps every accelerator wait behind one helper, so no
    ``torch.cuda.*`` call can run on an MPS machine.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


#: Highest spherical-harmonics degree mpsplat's Metal SH kernel supports (``K = (deg+1)^2 <= 25``).
#: ``mpsplat``'s ``gsplat/cuda/_wrapper.py:374-387`` routes ``spherical_harmonics`` to the Metal
#: kernel only while ``coeffs.shape[-2] <= 25``; a higher degree silently evaluates on the pure-torch
#: path instead, which is several times slower. Nothing else in ffsplat's rasterization call chain
#: has such a silent guard - see ``tests/test_metal_path.py``.
MAX_METAL_SH_DEGREE = 4


def warn_if_not_metal(sh_degree: int, device: torch.device) -> None:
    """Warn before a render that would silently leave mpsplat's Metal kernels.

    Only the SH degree can do this for ffsplat's argument set (pinhole camera, ``render_mode="RGB"``,
    no ``backgrounds``/``masks``, default ``tile_size``/``rasterize_mode``). The render stays correct
    on the fallback, it is just much slower, so this is a warning rather than an error.
    """
    if device.type == "mps" and sh_degree > MAX_METAL_SH_DEGREE:
        warnings.warn(
            f"sh_degree={sh_degree} is beyond mpsplat's Metal spherical-harmonics kernel "
            f"(degree <= {MAX_METAL_SH_DEGREE}); the spherical harmonics will be evaluated on the "
            f"slow pure-torch path.",
            RuntimeWarning,
            stacklevel=2,
        )
