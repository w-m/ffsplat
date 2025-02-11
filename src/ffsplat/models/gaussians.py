from dataclasses import dataclass
from typing import Optional

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from .attribute import NamedAttribute


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
        """Calculate spherical harmonics degree from the data shape."""
        return int(np.sqrt(self.sh.data.shape[1] - 1) - 1)  # -1 for sh0

    @property
    def sh0(self) -> Tensor:
        """Get the DC term of the spherical harmonics."""
        return self.sh.data[:, 0]

    @property
    def shN(self) -> Tensor:
        """Get the higher-order spherical harmonics coefficients."""
        return self.sh.data[:, 1:]

    def to(self, device: str) -> None:
        """Move all attributes to specified device in-place."""
        for attr in [self.means, self.quaternions, self.scales, self.opacities, self.sh]:
            attr.to(device)

    def to_torch(self, device: Optional[str] = None) -> tuple[Tensor, ...]:
        """Get all transformed attributes as PyTorch tensors."""
        if device is not None:
            self.to(device)

        sh_data = self.sh(None)  # Get transformed spherical harmonics data

        return (
            self.means(None),
            self.quaternions(None),
            self.scales(None),
            self.opacities(None),
            sh_data,  # Full spherical harmonics data
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
        """Create a Gaussians instance from numpy arrays."""
        # Combine sh0 and shN into a single array for storage
        sh_combined = np.concatenate([sh0, shN], axis=1)

        return cls(
            means=NamedAttribute.from_numpy("means", means, device),
            quaternions=NamedAttribute.from_numpy("quaternions", quats, device),
            # Scales use exp activation
            scales=NamedAttribute.from_numpy("scales", scales, device, remap_method="exp"),
            # Opacities use sigmoid activation
            opacities=NamedAttribute.from_numpy("opacities", opacities, device, remap_method="sigmoid"),
            # SH stored as concatenated data - no special transform needed
            sh=NamedAttribute.from_numpy("sh", sh_combined, device),
        )
