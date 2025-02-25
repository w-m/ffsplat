import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor


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
    means: Float[Tensor, "N 3"]
    quaternions: Float[Tensor, "N 4"]
    scales: Float[Tensor, "N 3"]
    opacities: Float[Tensor, " N"]
    sh: Float[Tensor, "N S 3"]

    def __str__(self) -> str:
        """Return a compact string representation of the Gaussians."""
        try:
            N = self.num_gaussians
            n_display = f"{N:,}"
        except ValueError:
            N = None
            n_display = "?"

        attrs = [
            f"{name}: {attr_tensor_str(attr, N)}"
            for name, attr in [
                ("means", self.means),
                ("quaternions", self.quaternions),
                ("scales", self.scales),
                ("opacities", self.opacities),
                ("sh", self.sh),
            ]
        ]

        return f"Gaussians(N={n_display}, sh_degree={self.sh_degree}, device={self.device})\n  " + "\n  ".join(attrs)

    @property
    def num_gaussians(self) -> int:
        """Get the number of gaussians from means attribute."""
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

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
        return int(math.sqrt(self.sh.shape[1]) - 1)
