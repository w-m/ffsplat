import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..models.fields import Field


def attr_tensor_str(data: Tensor, num_primitives: int | None = None) -> str:
    """Return a compact string representation of the attribute."""

    def format_shape(shape: tuple) -> str:
        dims = []
        for d in shape:
            if num_primitives is not None and d == num_primitives:
                dims.append("N")
            else:
                dims.append(str(d))
        return ", ".join(dims)

    return f"({format_shape(data.shape)}) [{data.min():.3g}, {data.max():.3g}]"


@dataclass
class Gaussians:
    means: Field
    quaternions: Field
    scales: Field
    opacities: Field
    sh: Field

    def __str__(self) -> str:
        """Return a compact string representation of the Gaussians."""
        try:
            N = self.num_gaussians
            n_display = f"{N:,}"
        except ValueError:
            N = None
            n_display = "?"

        attrs = [f"{name}: {attr_tensor_str(attr, N)}" for name, attr in self.to_dict().items()]

        return f"Gaussians(N={n_display}, sh_degree={self.sh_degree}, device={self.device})\n  " + "\n  ".join(attrs)

    @property
    def num_gaussians(self) -> int:
        """Get the number of gaussians from means attribute."""
        return self.means.data.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.data.device

    def to(self, device: str | torch.device) -> "Gaussians":
        return Gaussians(
            means=self.means.to(device),
            quaternions=self.quaternions.to(device),
            scales=self.scales.to(device),
            opacities=self.opacities.to(device),
            sh=self.sh.to(device),
        )

    @property
    def sh_degree(self) -> int:
        # TODO check this implementation
        # Data shape is (N, num_coeffs, 3), where num_coeffs = (degree + 1)^2
        # So num_coeffs = shape[1], and we solve for degree
        return int(math.sqrt(self.sh.data.shape[1]) - 1)

    def to_field_dict(self) -> dict:
        return {
            "means": self.means,
            "quaternions": self.quaternions,
            "scales": self.scales,
            "opacities": self.opacities,
            "sh": self.sh,
        }

    def to_dict(self) -> dict:
        return {
            "means": self.means.data,
            "quaternions": self.quaternions.data,
            "scales": self.scales.data,
            "opacities": self.opacities.data,
            "sh": self.sh.data,
        }
