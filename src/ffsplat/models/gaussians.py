from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from .attribute import NamedAttribute


@dataclass
class Gaussians:
    """Collection of 3D Gaussians with their attributes stored as torch tensors"""

    means_attr: NamedAttribute
    quaternions_attr: NamedAttribute
    scales_attr: NamedAttribute
    opacities_attr: NamedAttribute
    sh_attr: NamedAttribute  # Combined spherical harmonics (sh0 and shN)

    @property
    def device(self) -> torch.device:
        return self.means_attr.device

    @property
    def means(self) -> Tensor:
        return self.means_attr()

    @property
    def quaternions(self) -> Tensor:
        return self.quaternions_attr()

    @property
    def scales(self) -> Tensor:
        return self.scales_attr()

    @property
    def opacities(self) -> Tensor:
        return self.opacities_attr()

    @property
    def sh(self) -> Tensor:
        return self.sh_attr()

    @property
    def sh_degree(self) -> int:
        """Calculate spherical harmonics degree from the data shape."""
        return int(np.sqrt(self.sh_attr.raw_data.shape[1] - 1) - 1)  # -1 for sh0

    # @property
    # def sh0(self) -> Tensor:
    #     """Get the DC term of the spherical harmonics."""
    #     return self.sh_attr.data[:, 0]

    # @property
    # def shN(self) -> Tensor:
    #     """Get the higher-order spherical harmonics coefficients."""
    #     return self.sh_attr.data[:, 1:]

    def to(self, device) -> "Gaussians":
        """Create a new Gaussians instance with all attributes moved to the specified device."""
        if self.device == device:
            return self

        return Gaussians(
            means_attr=self.means_attr.to(device),
            quaternions_attr=self.quaternions_attr.to(device),
            scales_attr=self.scales_attr.to(device),
            opacities_attr=self.opacities_attr.to(device),
            sh_attr=self.sh_attr.to(device),
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
            means_attr=NamedAttribute.from_numpy("means", means, device),
            quaternions_attr=NamedAttribute.from_numpy("quaternions", quats, device),
            # Scales use exp activation
            scales_attr=NamedAttribute.from_numpy("scales", scales, device, remap_method="exp"),
            # Opacities use sigmoid activation
            opacities_attr=NamedAttribute.from_numpy("opacities", opacities, device, remap_method="sigmoid"),
            # SH stored as concatenated data - no special transform needed
            sh_attr=NamedAttribute.from_numpy("sh", sh_combined, device),
        )
