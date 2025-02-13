from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .attr_transforms import (
    BitShiftDecodingParams,
    BitShiftEncodingConfig,
    BitshiftTransform,
    ClampingEncodingConfig,
    ClampingTransform,
    CodebookLookupDecodingParams,
    CodebookLookupEncodingConfig,
    CodebookTransform,
    DTypeDecodingParams,
    DTypeEncodingConfig,
    DTypeTransform,
    EncodingTransform,
    LogicalOrTransform,
    QuantizationEncodingConfig,
    QuantizationTransform,
    RemappingDecodingParams,
    RemappingEncodingConfig,
    RemappingTransform,
    RescalingDecodingParams,
    RescalingEncodingConfig,
    RescalingTransform,
    SplitDecodingParams,
    SplitEncodingConfig,
    SplitTransform,
    SquareGridTransform,
    TrimmingEncodingConfig,
    TrimmingTransform,
)


@dataclass
class AttributeEncodingConfig:
    coding: dict[str, Any]
    reshaping: None
    trimming: TrimmingEncodingConfig | None
    coding_dtype: DTypeEncodingConfig | None
    quantization: QuantizationEncodingConfig | None
    bit_shift: BitShiftEncodingConfig | None
    codebook: CodebookLookupEncodingConfig | None
    split: SplitEncodingConfig | None
    logical_or: None
    combined_dtype: DTypeEncodingConfig | None
    rescaling: RescalingEncodingConfig | None
    clamping: ClampingEncodingConfig | None
    remapping: RemappingEncodingConfig | None


@dataclass
class AttributeDecodingParams:
    coding: dict[str, Any] = field(default_factory=dict)
    reshaping: None = None
    trimming: None = None
    coding_dtype: None = None
    quantization: None = None
    bit_shift: BitShiftDecodingParams | None = None
    codebook: CodebookLookupDecodingParams | None = None
    split: SplitDecodingParams | None = None
    logical_or: None = None
    combined_dtype: DTypeDecodingParams | None = None
    rescaling: RescalingDecodingParams | None = None
    clamping: None = None
    remapping: RemappingDecodingParams | None = None


class NamedAttribute:
    name: str

    # the underlying data, that can be stored in a buffer (e.g JPEG)
    _packed_data: Tensor | None
    # the parameters that are used to render the scene (e.g. linear scale values)
    _scene_params: Tensor | None

    # packed_data --> decode --> scene_params
    # scene_params --> encode --> packed_data

    # coding: EncodingTransform | None
    # reshaping: SquareGridTransform | None
    # trimming: TrimmingTransform | None
    # coding_dtype: DTypeTransform | None
    # quantization: QuantizationTransform | None
    # bit_shift: BitshiftTransform | None
    # codebook: CodebookTransform | None
    # split: SplitTransform | None
    # logical_or: LogicalOrTransform | None
    # combined_dtype: DTypeTransform | None
    # rescaling: RescalingTransform | None
    # clamping: ClampingTransform | None
    # remapping: RemappingTransform | None

    coding: EncodingTransform | None = None
    reshaping: EncodingTransform | None = None
    trimming: EncodingTransform | None = None
    coding_dtype: EncodingTransform | None = None
    quantization: EncodingTransform | None = None
    bit_shift: EncodingTransform | None = None
    codebook: EncodingTransform | None = None
    split: EncodingTransform | None = None
    logical_or: EncodingTransform | None = None
    combined_dtype: EncodingTransform | None = None
    rescaling: EncodingTransform | None = None
    clamping: EncodingTransform | None = None
    remapping: EncodingTransform | None = None

    def __init__(
        self,
        *,
        name: str,
        encoding_config: AttributeEncodingConfig | None = None,
        scene_params: Tensor | None = None,
        decoding_params: AttributeDecodingParams | None = None,
        packed_data: Tensor | None = None,
    ) -> None:
        super().__init__()

        if encoding_config and decoding_params:
            raise ValueError("Provide either encoding_config OR decoding_params, not both.")

        if not encoding_config and not decoding_params:
            raise ValueError("Provide either encoding_config OR decoding_params.")

        if encoding_config and scene_params is None:
            raise ValueError("Provide scene_params when using encoding_config.")

        if decoding_params and packed_data is None:
            raise ValueError("Provide packed_data when using decoding_params.")

        self.name = name

        if scene_params is not None:
            self._scene_params = scene_params
        if packed_data is not None:
            self._packed_data = packed_data

        # Initialize transforms for encoding path
        if encoding_config:
            self.coding = (
                EncodingTransform.from_encoding_config(encoding_config.coding) if encoding_config.coding else None
            )
            self.reshaping = SquareGridTransform() if encoding_config.reshaping else None
            self.trimming = (
                TrimmingTransform.from_encoding_config(encoding_config.trimming) if encoding_config.trimming else None
            )
            self.coding_dtype = (
                DTypeTransform.from_encoding_config(encoding_config.coding_dtype)
                if encoding_config.coding_dtype
                else None
            )
            self.quantization = (
                QuantizationTransform.from_encoding_config(encoding_config.quantization)
                if encoding_config.quantization
                else None
            )
            self.bit_shift = (
                BitshiftTransform.from_encoding_config(encoding_config.bit_shift) if encoding_config.bit_shift else None
            )
            self.codebook = (
                CodebookTransform.from_encoding_config(encoding_config.codebook) if encoding_config.codebook else None
            )
            self.split = SplitTransform.from_encoding_config(encoding_config.split) if encoding_config.split else None
            self.logical_or = LogicalOrTransform() if encoding_config.logical_or else None
            self.combined_dtype = (
                DTypeTransform.from_encoding_config(encoding_config.combined_dtype)
                if encoding_config.combined_dtype
                else None
            )
            self.rescaling = (
                RescalingTransform.from_encoding_config(encoding_config.rescaling)
                if encoding_config.rescaling
                else None
            )
            self.clamping = (
                ClampingTransform.from_encoding_config(encoding_config.clamping) if encoding_config.clamping else None
            )
            self.remapping = (
                RemappingTransform.from_encoding_config(encoding_config.remapping)
                if encoding_config.remapping
                else None
            )

        # Initialize transforms for decoding path
        elif decoding_params:
            self.coding = (
                EncodingTransform.from_decoding_params(decoding_params.coding) if decoding_params.coding else None
            )
            self.reshaping = SquareGridTransform() if decoding_params.reshaping else None
            self.trimming = None  # Trimming is encode-only
            self.coding_dtype = None  # No params needed for decode
            self.quantization = None  # Quantization is encode-only
            self.bit_shift = (
                BitshiftTransform.from_decoding_params(decoding_params.bit_shift) if decoding_params.bit_shift else None
            )
            self.codebook = (
                CodebookTransform.from_decoding_params(decoding_params.codebook) if decoding_params.codebook else None
            )
            self.split = SplitTransform.from_decoding_params(decoding_params.split) if decoding_params.split else None
            self.logical_or = LogicalOrTransform() if decoding_params.logical_or else None
            self.combined_dtype = (
                DTypeTransform.from_decoding_params(decoding_params.combined_dtype)
                if decoding_params.combined_dtype
                else None
            )
            self.rescaling = (
                RescalingTransform.from_decoding_params(decoding_params.rescaling)
                if decoding_params.rescaling
                else None
            )
            self.clamping = None  # Clamping is encode-only
            self.remapping = (
                RemappingTransform.from_decoding_params(decoding_params.remapping)
                if decoding_params.remapping
                else None
            )

        self._transforms = [
            t
            for t in [
                self.remapping,
                self.clamping,
                self.rescaling,
                self.combined_dtype,
                self.logical_or,
                self.split,
                self.codebook,
                self.bit_shift,
                self.quantization,
                self.coding_dtype,
                self.trimming,
                self.reshaping,
                self.coding,
            ]
            if t is not None
        ]

    @property
    def device(self) -> torch.device:
        """Get the device of the data. Prefer scene_params if available."""
        if hasattr(self, "scene_params") and self._scene_params is not None:
            return self._scene_params.device
        if hasattr(self, "packed_data") and self._packed_data is not None:
            return self._packed_data.device
        raise ValueError("No data available to determine device")

    @property
    def scene_params(self) -> Tensor:
        """Get the scene parameters tensor."""
        if self._scene_params is None:
            raise ValueError("No scene_params available. Run decode()?")
        return self._scene_params

    @property
    def packed_data(self) -> Tensor:
        """Get the packed data tensor."""
        if self._packed_data is None:
            raise ValueError("No packed_data available. Run encode()?")
        return self._packed_data

    def to(self, device: torch.device | str) -> NamedAttribute:
        """Move all data and transforms to the specified device."""
        device = torch.device(device)  # Convert string to device if needed

        if hasattr(self, "scene_params") and self._scene_params is not None:
            self._scene_params = self._scene_params.to(device)
        if hasattr(self, "packed_data") and self._packed_data is not None:
            self._packed_data = self._packed_data.to(device)

        # Move all transforms that need device movement
        for transform in [self.coding, self.codebook, self.split]:
            if transform is not None:
                transform.to(device)
        return self

    def encode(self) -> Tensor:
        """Apply the encoding pipeline to scene_params to produce packed_data."""
        if not hasattr(self, "scene_params") or self._scene_params is None:
            raise ValueError("No scene_params available for encoding")

        data = self._scene_params
        for transform in self._transforms:
            data = transform.encode(data)

        self._packed_data = data
        return data

    def decode(self) -> Tensor:
        """Apply the decoding pipeline to packed_data to reconstruct scene_params."""

        if not hasattr(self, "packed_data") or self._packed_data is None:
            raise ValueError("No packed_data available for decoding")

        data = self._packed_data
        for transform in reversed(self._transforms):
            data = transform.decode(data)

        self._scene_params = data
        return data

    @classmethod
    def from_scene_params(
        cls,
        *,
        name: str,
        scene_params: Tensor | NDArray,
        remapping: Literal["exp", "sigmoid"] | None = None,
        clamping: tuple[float | None, float | None] | None = None,
        rescaling: tuple[float, float] | None = None,
        combined_dtype: torch.dtype | None = None,
        logical_or: bool = False,
        split: tuple[int, int] | None = None,
        codebook: NamedAttribute | None = None,
        bit_shift: int | None = None,
        quantization: int | None = None,
        coding_dtype: torch.dtype | None = None,
        trimming: int | None = None,
        reshaping: bool = False,
        coding: dict[str, Any] | None = None,
    ) -> Self:
        """Create a NamedAttribute configured for encoding from scene_params to packed_data."""
        config = AttributeEncodingConfig(
            remapping=RemappingEncodingConfig(method=remapping) if remapping else None,
            clamping=ClampingEncodingConfig(min_val=clamping[0], max_val=clamping[1]) if clamping else None,
            rescaling=RescalingEncodingConfig(target_min=rescaling[0], target_max=rescaling[1]) if rescaling else None,
            combined_dtype=DTypeEncodingConfig(dtype=combined_dtype) if combined_dtype else None,
            logical_or=None,  # Just pass None since it's a flag
            split=SplitEncodingConfig(split_dim=split[0], split_size=split[1]) if split else None,
            codebook=CodebookLookupEncodingConfig(codebook=codebook) if codebook else None,
            bit_shift=BitShiftEncodingConfig(right_shift=bit_shift) if bit_shift else None,
            quantization=QuantizationEncodingConfig(num_bits=quantization) if quantization else None,
            coding_dtype=DTypeEncodingConfig(dtype=coding_dtype) if coding_dtype else None,
            trimming=TrimmingEncodingConfig(num_elements_to_trim=trimming) if trimming else None,
            reshaping=None,  # Just pass None since it's a flag
            coding=coding or {},
        )

        if isinstance(scene_params, np.ndarray):
            scene_params = torch.from_numpy(scene_params)

        # Initialize with transforms based on flags
        result = cls(name=name, encoding_config=config, scene_params=scene_params)
        if logical_or:
            result.logical_or = LogicalOrTransform()
        if reshaping:
            result.reshaping = SquareGridTransform()
        return result

    @classmethod
    def from_packed_data(
        cls,
        *,
        name: str,
        packed_data: Tensor | NDArray,
        remapping: Literal["exp", "sigmoid"] | None = None,
        rescaling: tuple[float, float, float, float] | None = None,
        combined_dtype: tuple[torch.dtype, torch.dtype] | None = None,
        logical_or: bool = False,
        split: tuple[int, list[NamedAttribute]] | None = None,
        codebook: NamedAttribute | None = None,
        bit_shift: int | None = None,
        reshaping: bool = False,
        coding: dict[str, Any] | None = None,
    ) -> Self:
        """Create a NamedAttribute configured for decoding from packed_data to scene_params.

        Args:
            name: Name of the attribute
            packed_data: Data to decode, can be either a torch.Tensor or a numpy.ndarray
            remapping: Optional remapping method to apply during decode
            rescaling: Optional (orig_min, orig_max, target_min, target_max) for rescaling
            combined_dtype: Optional (orig_dtype, target_dtype) for dtype conversion
            logical_or: Whether to apply logical OR transform
            split: Optional (concat_dim, attributes) for split/concat
            codebook: Optional codebook for lookup
            bit_shift: Optional number of bits to left shift
            reshaping: Whether to reshape to/from square grid
            coding: Optional coding parameters
        """
        # Convert numpy array to tensor if needed
        if isinstance(packed_data, np.ndarray):
            packed_data = torch.from_numpy(packed_data)

        params = AttributeDecodingParams(
            remapping=RemappingDecodingParams(method=remapping) if remapping else None,
            rescaling=RescalingDecodingParams(
                orig_min=rescaling[0], orig_max=rescaling[1], target_min=rescaling[2], target_max=rescaling[3]
            )
            if rescaling
            else None,
            combined_dtype=DTypeDecodingParams(orig_dtype=combined_dtype[0], target_dtype=combined_dtype[1])
            if combined_dtype
            else None,
            logical_or=None,  # Just pass None since it's a flag
            split=SplitDecodingParams(concat_dim=split[0], attributes=split[1]) if split else None,
            codebook=CodebookLookupDecodingParams(codebook=codebook) if codebook else None,
            bit_shift=BitShiftDecodingParams(left_shift=bit_shift) if bit_shift else None,
            reshaping=None,  # Just pass None since it's a flag
            coding=coding or {},
        )
        # Initialize with transforms based on flags
        result = cls(name=name, decoding_params=params, packed_data=packed_data)
        if logical_or:
            result.logical_or = LogicalOrTransform()
        if reshaping:
            result.reshaping = SquareGridTransform()
        return result
