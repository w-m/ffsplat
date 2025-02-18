from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from .field import (
    FieldEncodingConfig,
    NamedField,
    RemappingEncodingConfig,
)


@dataclass
class GaussiansEncodingConfig:
    """Holds encoding configurations for all attributes of a Gaussians instance"""

    means: FieldEncodingConfig = field(default_factory=FieldEncodingConfig)
    quaternions: FieldEncodingConfig = field(default_factory=FieldEncodingConfig)
    scales: FieldEncodingConfig = field(default_factory=FieldEncodingConfig)
    opacities: FieldEncodingConfig = field(default_factory=FieldEncodingConfig)
    sh: FieldEncodingConfig = field(default_factory=FieldEncodingConfig)
    # TODO how to deal with dynamic field names?
    # __dict__: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gaussians:
    """Collection of 3D Gaussians with their attributes stored as torch tensors"""

    means_attr: NamedField
    quaternions_attr: NamedField
    scales_attr: NamedField
    opacities_attr: NamedField
    sh_attr: NamedField  # Combined spherical harmonics (sh0 and shN)

    @property
    def num_gaussians(self) -> int:
        """Get the number of gaussians from means attribute."""
        return self.means_attr.num_gaussians

    @property
    def device(self) -> torch.device:
        return self.means_attr.device

    def to(self, device: str | torch.device) -> "Gaussians":
        """Move all attributes to the specified device."""
        self.means_attr.to(device)
        self.quaternions_attr.to(device)
        self.scales_attr.to(device)
        self.opacities_attr.to(device)
        self.sh_attr.to(device)
        return self

    def __str__(self) -> str:
        """Return a compact string representation of the Gaussians."""
        try:
            N = self.num_gaussians
            n_display = f"{N:,}"
        except ValueError:
            N = None
            n_display = "?"

        attrs = [
            attr.__str__(N)
            for attr in [self.means_attr, self.quaternions_attr, self.scales_attr, self.opacities_attr, self.sh_attr]
        ]

        return f"Gaussians(N={n_display}, device={self.device})\n  " + "\n  ".join(attrs)

    @property
    def sh_degree(self) -> int:
        """Calculate spherical harmonics degree from the data shape."""
        if self.sh_attr._scene_params is None and self.sh_attr._packed_data is None:
            raise ValueError("Neither scene_params nor packed_data available")

        # Use whichever data is available
        data = self.sh_attr._scene_params if self.sh_attr._scene_params is not None else self.sh_attr._packed_data
        if data is None:  # Purely for type checking - we know this can't happen
            raise ValueError("Internal error: data is None despite prior check")

        # Data shape is (N, num_coeffs, 3), where num_coeffs = (degree + 1)^2
        # So num_coeffs = shape[1], and we solve for degree
        return int(np.sqrt(data.shape[1]) - 1)

    def decode(self) -> None:
        """Decode all attributes."""
        self.means_attr.decode()
        self.quaternions_attr.decode()
        self.scales_attr.decode()
        self.opacities_attr.decode()
        self.sh_attr.decode()

    def encode(self) -> None:
        """Encode all attributes."""
        self.means_attr.encode()
        self.quaternions_attr.encode()
        self.scales_attr.encode()
        self.opacities_attr.encode()
        self.sh_attr.encode()

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

        def create_attribute(
            name: str, data: NDArray, remap_method: Literal["exp", "sigmoid"] | None = None
        ) -> NamedField:
            tensor = torch.from_numpy(data).to(device)
            config = None
            if remap_method:
                config = FieldEncodingConfig(
                    coding={},  # No coding by default
                    reshaping=None,
                    trimming=None,
                    coding_dtype=None,
                    quantization=None,
                    bit_shift=None,
                    codebook=None,
                    split=None,
                    logical_or=None,
                    combined_dtype=None,
                    rescaling=None,
                    clamping=None,
                    remapping=RemappingEncodingConfig(method=remap_method),
                )
            return NamedField(name=name, scene_params=tensor, encoding_config=config)

        return cls(
            means_attr=create_attribute("means", means),
            quaternions_attr=create_attribute("quaternions", quats),
            scales_attr=create_attribute("scales", scales, remap_method="exp"),
            opacities_attr=create_attribute("opacities", opacities, remap_method="sigmoid"),
            sh_attr=create_attribute("sh", sh_combined),
        )

    @classmethod
    def from_gaussians(cls, other: "Gaussians", encoding_config: dict) -> "Gaussians":
        """Create a new Gaussians instance from another, applying the specified encoding configs."""
        return cls(
            means_attr=NamedField(
                name="means", scene_params=other.means_attr.scene_params, encoding_config=encoding_config["means"]
            ),
            quaternions_attr=NamedField(
                name="quaternions",
                scene_params=other.quaternions_attr.scene_params,
                encoding_config=encoding_config["quaternions"],
            ),
            scales_attr=NamedField(
                name="scales", scene_params=other.scales_attr.scene_params, encoding_config=encoding_config["scales"]
            ),
            opacities_attr=NamedField(
                name="opacities",
                scene_params=other.opacities_attr.scene_params,
                encoding_config=encoding_config["opacities"],
            ),
            sh_attr=NamedField(
                name="sh", scene_params=other.sh_attr.scene_params, encoding_config=encoding_config["sh"]
            ),
        )


@dataclass
class GaussianSceneParams:
    means: Float[Tensor, "N 3"]
    quaternions: Float[Tensor, "N 4"]
    scales: Float[Tensor, "N 3"]
    opacities: Float[Tensor, " N"]
    sh: Float[Tensor, "N S 3"]
