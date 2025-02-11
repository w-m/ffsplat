from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor


@dataclass
class NamedAttribute:
    """Generic attribute class that handles Gaussian attributes stored as torch tensors"""

    name: str
    data: Tensor
    device: str = "cuda"
    to_device_fn: Callable[[Tensor, str], Tensor] = lambda x, d: x.to(d)

    @classmethod
    def from_numpy(
        cls,
        name: str,
        data: NDArray,
        device: str = "cuda",
        to_device_fn: Optional[Callable[[Tensor, str], Tensor]] = None,
    ) -> "NamedAttribute":
        """Create a NamedAttribute from a numpy array, converting to torch tensor"""
        tensor = torch.from_numpy(data)
        if to_device_fn is None:
            to_device_fn = cls.to_device_fn
        return cls(name, to_device_fn(tensor, device), device, to_device_fn)

    def to(self, device: str) -> None:
        """Move the tensor to specified device in-place"""
        if device != self.device:
            self.data = self.to_device_fn(self.data, device)
            self.device = device


@dataclass
class Gaussians:
    """Collection of 3D Gaussians with their attributes stored as torch tensors"""

    means: NamedAttribute
    quaternions: NamedAttribute
    scales: NamedAttribute
    opacities: NamedAttribute
    sh: NamedAttribute  # Combined spherical harmonics (sh0 and shN)

    @property
    def device(self) -> str:
        return self.means.device

    @property
    def sh_degree(self) -> int:
        """Calculate spherical harmonics degree from the data shape"""
        return int(np.sqrt(self.sh.data.shape[1] - 1) - 1)  # -1 for sh0

    @property
    def sh0(self) -> Tensor:
        """Get the DC term (sh0) of spherical harmonics"""
        return self.sh.data[:, :1, :]

    @property
    def shN(self) -> Tensor:
        """Get the higher-order terms (shN) of spherical harmonics"""
        return self.sh.data[:, 1:, :]

    def to(self, device: str) -> None:
        """Move all attributes to specified device in-place"""
        for attr in [self.means, self.quaternions, self.scales, self.opacities, self.sh]:
            attr.to(device)

    def to_torch(self, device: Optional[str] = None) -> tuple[Tensor, ...]:
        """Get all attributes as PyTorch tensors"""
        if device is not None:
            self.to(device)

        return (
            self.means.data,
            self.quaternions.data,
            self.scales.data,
            self.opacities.data,
            self.sh.data,
        )

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
        device: str = "cuda",
    ) -> "Gaussians":
        """Create a Gaussians instance from numpy arrays"""
        # Combine sh0 and shN into a single array
        sh_combined = np.concatenate([sh0, shN], axis=1)

        return cls(
            means=NamedAttribute.from_numpy("means", means, device),
            quaternions=NamedAttribute.from_numpy("quaternions", quats, device),
            scales=NamedAttribute.from_numpy("scales", scales, device, lambda x, d: torch.exp(x.to(d))),
            opacities=NamedAttribute.from_numpy("opacities", opacities, device, lambda x, d: torch.sigmoid(x.to(d))),
            sh=NamedAttribute.from_numpy("sh", sh_combined, device),
        )
