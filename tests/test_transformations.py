"""Basic unit tests for a subset of helper functions in
``ffsplat.models.transformations``.

"The goal is **not** to provide exhaustive coverage - only to ensure that the
test-suite infrastructure is wired up and at least one realistic test is
executed successfully.  To avoid introducing heavyweight binary dependencies
that may not be available in the execution environment (such as *opencv* or
*torchpq*), we monkey-patch a few optional third-party modules *before* the
target module is imported.
"""

from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------
# Optional run-time dependencies that are imported at module scope inside
# ``ffsplat.models.transformations``.  We only need dummy placeholders so that
# ``import`` succeeds; the functionality under test (``minmax`` and
# ``convert_to_dtype``) does **not** rely on these libraries.
# ---------------------------------------------------------------------------

# Include a small set of heavy optional dependencies that we replace with
# stubs when they are not available in the execution environment.  *Only* those
# that are imported at module scope by ``transformations`` need to be included.
_DUMMY_MODULES = [
    "cv2",
    "pillow_heif",
    "plas",
    "PIL",  # Make ``import PIL`` work.
]

# Create a one-line stub for each missing module.
for _name in _DUMMY_MODULES:
    if _name not in sys.modules:
        stub = types.ModuleType(_name)
        # The transformations module does ``from pillow_heif import register_avif_opener``.
        # Provide a dummy callable so that the import succeeds when the real
        # library is absent.
        if _name == "pillow_heif":
            stub.register_avif_opener = lambda *_, **__: None  # type: ignore[attr-defined]
        elif _name == "plas":
            # Provide a minimal ``sort_with_plas`` stub.
            stub.sort_with_plas = lambda *_, **__: None  # type: ignore[attr-defined]
        elif _name == "cv2":
            # stub the ``imwrite`` function used by ``write_image``.
            stub.imwrite = lambda *_, **__: True  # type: ignore[attr-defined]
        sys.modules[_name] = stub

# Special handling for the ``PIL`` namespace so that ``from PIL import Image``
# works without the real *Pillow* package being installed.
if "PIL" in sys.modules:
    pil_module = sys.modules["PIL"]
    if not hasattr(pil_module, "Image"):
        image_stub = types.ModuleType("PIL.Image")
        pil_module.Image = image_stub  # type: ignore[attr-defined]
        sys.modules["PIL.Image"] = image_stub


# ``ffsplat.models.transformations`` performs
# ``from torchpq.clustering import KMeans`` at import time. We therefore need a
# synthetic sub-module that exposes a dummy ``KMeans`` symbol so that the import
# statement does not fail if *torchpq* is not installed.

if "torchpq.clustering" not in sys.modules:
    clustering_stub = types.ModuleType("torchpq.clustering")

    class _DummyKMeans:
        """Very small stub replacing :class:`torchpq.clustering.KMeans`."""

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return []

        @property
        def centroids(self):
            import torch

            # Just return a zero tensor with a plausible shape so that attribute
            # access in client code will not fail.
            return torch.zeros((1, 1))

    clustering_stub.KMeans = _DummyKMeans  # type: ignore[attr-defined]

    # Register both the top-level ``torchpq`` *and* the sub-module so relative
    # imports work as expected.
    torchpq_stub = types.ModuleType("torchpq")
    torchpq_stub.clustering = clustering_stub  # type: ignore[attr-defined]

    sys.modules["torchpq"] = torchpq_stub
    sys.modules["torchpq.clustering"] = clustering_stub


# Now we can safely import the module under test.
# Skip the whole test module if *PyTorch* is not available.  We also fall back
# to a very small stub in case *pytest* itself is missing in the current
# execution environment (this should never happen in the official test runner,
# but allows the file to be imported elsewhere without hard dependencies).

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs.
    pytest = types.ModuleType("pytest")  # type: ignore[assignment]

    def _importorskip(name: str, *_, **__) -> types.ModuleType:  # type: ignore[return-type]
        try:
            return __import__(name)
        except ModuleNotFoundError:  # pragma: no cover
            # Return a dummy placeholder - the calling test will likely fail
            # later, but at least the import of *this* module succeeds.
            dummy = types.ModuleType(name)
            sys.modules[name] = dummy
            return dummy

    pytest.importorskip = _importorskip  # type: ignore[attr-defined]
    sys.modules["pytest"] = pytest


torch = pytest.importorskip("torch")  # type: ignore[assignment]

from ffsplat.models.transformations import convert_to_dtype, minmax  # noqa: E402

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
