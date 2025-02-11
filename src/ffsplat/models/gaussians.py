from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor


@dataclass
class NamedAttribute:
    """Generic attribute class that can handle different types of Gaussian attributes"""

    name: str
    data: NDArray
    to_torch_fn: Callable[[NDArray, str], Tensor] = lambda x, d: torch.from_numpy(x).to(d)

    def to_torch(self, device: str = "cuda") -> Tensor:
        return self.to_torch_fn(self.data, device)

    @classmethod
    def from_numpy(
        cls, name: str, data: NDArray, to_torch_fn: Optional[Callable[[NDArray, str], Tensor]] = None
    ) -> "NamedAttribute":
        return cls(name, data, to_torch_fn or cls.to_torch_fn)


@dataclass
class Gaussians:
    """Collection of 3D Gaussians with their attributes"""

    means: NamedAttribute
    quaternions: NamedAttribute
    scales: NamedAttribute
    opacities: NamedAttribute
    harmonics: tuple[NamedAttribute, NamedAttribute]  # (sh0, shN)
    _device: Optional[str] = None

    @property
    def device(self) -> str:
        return self._device or "cuda"

    @property
    def sh_degree(self) -> int:
        """Calculate spherical harmonics degree from the data shape"""
        return int(np.sqrt(self.harmonics[1].data.shape[1] + 1) - 1)

    def to_torch(self, device: Optional[str] = None) -> tuple[Tensor, ...]:
        """Convert all attributes to PyTorch tensors"""
        if device is not None:
            self._device = device

        means_t = self.means.to_torch(self.device)
        quats_t = self.quaternions.to_torch(self.device)
        scales_t = self.scales.to_torch(self.device)
        opacities_t = self.opacities.to_torch(self.device)
        sh0_t = self.harmonics[0].to_torch(self.device)
        shN_t = self.harmonics[1].to_torch(self.device)
        colors_t = torch.cat([sh0_t, shN_t], dim=-2)

        return means_t, quats_t, scales_t, opacities_t, colors_t

    @classmethod
    def from_numpy(
        cls,
        *,  # Force keyword arguments
        means: Float[NDArray, "N 3"],
        quats: Float[NDArray, "N 4"],
        scales: Float[NDArray, "N 3"],
        opacities: Float[NDArray, " N"],
        sh0: Float[NDArray, "N 1 3"],
        shN: Float[NDArray, "N S 3"],
    ) -> "Gaussians":
        """Create a Gaussians instance from numpy arrays"""
        return cls(
            means=NamedAttribute.from_numpy("means", means),
            quaternions=NamedAttribute.from_numpy("quaternions", quats),
            scales=NamedAttribute.from_numpy("scales", scales, lambda x, d: torch.exp(torch.from_numpy(x).to(d))),
            opacities=NamedAttribute.from_numpy(
                "opacities", opacities, lambda x, d: torch.sigmoid(torch.from_numpy(x).to(d))
            ),
            harmonics=(NamedAttribute.from_numpy("sh0", sh0), NamedAttribute.from_numpy("shN", shN)),
        )
