from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Self, TypeVar

import torch
from torch import Tensor

E = TypeVar("E")  # encoding config type
D = TypeVar("D")  # decoding params type


class EncodingTransform(Generic[E, D]):
    """
    Base class that can be initialized with encoding_config or decoding_params.
    Subclasses define encode/decode logic. This class provides easy getters and
    also ensures we don't set both config and params simultaneously.
    """

    def __init__(self, *, encoding_config: E | None = None, decoding_params: D | None = None) -> None:
        super().__init__()
        if encoding_config and decoding_params:
            raise ValueError("Provide either encoding_config OR decoding_params, not both.")
        self._encoding_config: E | None = encoding_config
        self._decoding_params: D | None = decoding_params

    @property
    def encoding_config(self) -> E | None:
        return self._encoding_config

    @property
    def decoding_params(self) -> D | None:
        return self._decoding_params

    def _encode_impl(self, data: Tensor, config: E) -> Tensor:
        """Implementation of encode logic. Override this in subclasses."""
        return data

    def _decode_impl(self, data: Tensor, params: D) -> Tensor:
        """Implementation of decode logic. Override this in subclasses."""
        return data

    def encode(self, data: Tensor) -> Tensor:
        """Public encode method that handles parameter validation."""
        if not self.encoding_config:
            raise ValueError("No encoding_config set; cannot encode.")
        return self._encode_impl(data, self.encoding_config)

    def decode(self, data: Tensor) -> Tensor:
        """Public decode method that handles parameter validation."""
        if not self.decoding_params:
            raise ValueError("No decoding_params set; cannot decode.")
        return self._decode_impl(data, self.decoding_params)

    def to(self, device: torch.device) -> Self:
        """Move all dependent tensors to the specified device."""
        return self

    @classmethod
    def from_encoding_config(cls, encoding_config: E) -> EncodingTransform[E, D]:
        return cls(encoding_config=encoding_config)

    @classmethod
    def from_decoding_params(cls, decoding_params: D) -> EncodingTransform[E, D]:
        return cls(decoding_params=decoding_params)


@dataclass
class QuantizationEncodingConfig:
    num_bits: int

    def __post_init__(self) -> None:
        if self.num_bits <= 0:
            raise ValueError("num_bits must be positive")


class QuantizationTransform(EncodingTransform[QuantizationEncodingConfig, None]):
    def _encode_impl(self, data: Tensor, config: QuantizationEncodingConfig) -> Tensor:
        levels = 2**config.num_bits - 1
        rounded: Tensor = torch.round(data * levels) / levels
        return rounded


@dataclass
class TrimmingEncodingConfig:
    num_elements_to_trim: int


class TrimmingTransform(EncodingTransform[TrimmingEncodingConfig, None]):
    """
    Removes the last N elements from the data at encode time.
    """

    def _encode_impl(self, data: Tensor, config: TrimmingEncodingConfig) -> Tensor:
        n = config.num_elements_to_trim
        if n <= 0 or n > data.shape[0]:
            raise ValueError(f"Invalid trimming length: {n} with data shape {data.shape}")
        return data[:-n]


@dataclass
class ClampingEncodingConfig:
    min_val: float | None
    max_val: float | None


class ClampingTransform(EncodingTransform[ClampingEncodingConfig, None]):
    def _encode_impl(self, data: Tensor, config: ClampingEncodingConfig) -> Tensor:
        return torch.clamp(data, min=config.min_val, max=config.max_val)


@dataclass
class RescalingEncodingConfig:
    target_min: float
    target_max: float


@dataclass
class RescalingDecodingParams:
    orig_min: float
    orig_max: float
    target_min: float
    target_max: float


class RescalingTransform(EncodingTransform[RescalingEncodingConfig, RescalingDecodingParams]):
    """
    Scales data from [orig_min, orig_max] to [target_min, target_max], or vice versa.
    """

    def _encode_impl(self, data: Tensor, config: RescalingEncodingConfig) -> Tensor:
        tmin = config.target_min
        tmax = config.target_max

        orig_min = float(data.min().item())
        orig_max = float(data.max().item())
        spread = (orig_max - orig_min) or 1.0

        scaled = (data - orig_min) / spread
        scaled = scaled * (tmax - tmin) + tmin

        self._decoding_params = RescalingDecodingParams(
            orig_min=orig_min,
            orig_max=orig_max,
            target_min=tmin,
            target_max=tmax,
        )
        return scaled

    def _decode_impl(self, data: Tensor, params: RescalingDecodingParams) -> Tensor:
        spread = (params.target_max - params.target_min) or 1.0
        unscaled = (data - params.target_min) / spread
        unscaled = unscaled * (params.orig_max - params.orig_min) + params.orig_min
        return unscaled


@dataclass
class BitShiftEncodingConfig:
    right_shift: int

    def __post_init__(self) -> None:
        if self.right_shift < 0:
            raise ValueError("right_shift must be positive")


@dataclass
class BitShiftDecodingParams:
    left_shift: int

    def __post_init__(self) -> None:
        if self.left_shift < 0:
            raise ValueError("left_shift must be positive")


class BitshiftTransform(EncodingTransform[BitShiftEncodingConfig, BitShiftDecodingParams]):
    def _encode_impl(self, data: Tensor, config: BitShiftEncodingConfig) -> Tensor:
        if torch.is_floating_point(data):
            raise ValueError("Bitshift transform is only for integer data.")

        shift_amount = config.right_shift
        self._decoding_params = BitShiftDecodingParams(left_shift=shift_amount)

        return data >> shift_amount

    def _decode_impl(self, data: Tensor, params: BitShiftDecodingParams) -> Tensor:
        if torch.is_floating_point(data):
            raise ValueError("Bitshift transform is only for integer data.")

        return data << params.left_shift


@dataclass
class DTypeEncodingConfig:
    dtype: torch.dtype


@dataclass
class DTypeDecodingParams:
    orig_dtype: torch.dtype
    target_dtype: torch.dtype


class DTypeTransform(EncodingTransform[DTypeEncodingConfig, DTypeDecodingParams]):
    def _encode_impl(self, data: Tensor, config: DTypeEncodingConfig) -> Tensor:
        orig_dtype = data.dtype
        self._decoding_params = DTypeDecodingParams(orig_dtype=orig_dtype, target_dtype=config.dtype)
        return data.to(config.dtype)

    def _decode_impl(self, data: Tensor, params: DTypeDecodingParams) -> Tensor:
        return data.to(params.orig_dtype)


@dataclass
class RemappingEncodingConfig:
    """Using the 'activation' naming of the 3DGS paper here. To encode,
    we need to use the inverse of these methods"""

    method: Literal["exp", "sigmoid"]


@dataclass
class RemappingDecodingParams:
    method: Literal["exp", "sigmoid"]


class RemappingTransform(EncodingTransform[RemappingEncodingConfig, RemappingDecodingParams]):
    def _encode_impl(self, data: Tensor, config: RemappingEncodingConfig) -> Tensor:
        match config.method:
            case "exp":
                return torch.log(data)
            case "sigmoid":
                return torch.log(data / (1 - data))
            case _:
                raise ValueError(f"Unknown remapping method: {self.method}")

    def _decode_impl(self, data: Tensor, params: RemappingDecodingParams) -> Tensor:
        match params.method:
            case "exp":
                return torch.exp(data)
            case "sigmoid":
                return torch.sigmoid(data)
            case _:
                raise ValueError(f"Unknown remapping method: {params.method}")


class SquareGridTransform(EncodingTransform[None, None]):
    def _encode_impl(self, data: Tensor, config: None) -> Tensor:
        num_elements = data.shape[0]
        square_size = int(math.sqrt(num_elements))
        if square_size**2 != num_elements:
            raise ValueError(
                f"GridTransform requires a square input, got {num_elements} elements. Trim to square shape!"
            )
        return data.reshape(square_size, square_size, *data.shape[1:])

    def _decode_impl(self, data: Tensor, params: None) -> Tensor:
        return torch.flatten(data, start_dim=0, end_dim=1)


@dataclass
class CodebookLookupEncodingConfig:
    codebook: NamedAttribute


@dataclass
class CodebookLookupDecodingParams:
    codebook: NamedAttribute


class CodebookTransform(EncodingTransform[CodebookLookupEncodingConfig, CodebookLookupDecodingParams]):
    def _encode_impl(self, data: Tensor, config: CodebookLookupEncodingConfig) -> Tensor:
        self._decoding_params = CodebookLookupDecodingParams(codebook=config.codebook)
        raise NotImplementedError
        return data

    def _decode_impl(self, data: Tensor, params: CodebookLookupDecodingParams) -> Tensor:
        raise NotImplementedError
        return params.codebook()[data]

    def to(self, device: torch.device) -> Self:
        if self._decoding_params:
            codebook = self._decoding_params.codebook.to(device)
        raise NotImplementedError
        return CodebookTransform(
            decoding_params=CodebookLookupDecodingParams(codebook=codebook), encoding_config=self.encoding_config
        )


@dataclass
class SplitEncodingConfig:
    split_dim: int
    split_size: int


@dataclass
class SplitDecodingParams:
    concat_dim: int
    attributes: list[NamedAttribute]


class SplitTransform(EncodingTransform[SplitEncodingConfig, SplitDecodingParams]):
    def _encode_impl(self, data: Tensor, config: SplitEncodingConfig) -> Tensor:
        split_dim = config.split_dim
        split_size = config.split_size
        if data.shape[split_dim] % split_size != 0:
            raise ValueError(
                f"Split size {split_size} does not divide the input shape {data.shape} along dimension {split_dim}"
            )
        # chunks_t = data.split(split_size, dim=split_dim)
        # build list of NamedAttributes
        # TODO
        raise NotImplementedError

    def _decode_impl(self, data: Tensor, params: SplitDecodingParams) -> Tensor:
        # TODO implement
        raise NotImplementedError
        chunks = [attr() for attr in params.attributes]
        return torch.cat([data, *chunks], dim=params.concat_dim)

    def to(self, device: torch.device) -> SplitTransform:
        raise NotImplementedError


class LogicalOrTransform(EncodingTransform[None, None]):
    """For combining hi/lo bytes into a single int"""

    def _encode_impl(self, data: Tensor, config: None) -> Tensor:
        raise NotImplementedError

    def _decode_impl(self, data: Tensor, params: None) -> Tensor:
        raise NotImplementedError

    def to(self, device: torch.device) -> LogicalOrTransform:
        raise NotImplementedError


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
    packed_data: Tensor | None
    # the parameters that are used to render the scene (e.g. linear scale values)
    scene_params: Tensor | None

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
            self.scene_params = scene_params
        if packed_data is not None:
            self.packed_data = packed_data

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
        if hasattr(self, "scene_params") and self.scene_params is not None:
            return self.scene_params.device
        if hasattr(self, "packed_data") and self.packed_data is not None:
            return self.packed_data.device
        raise ValueError("No data available to determine device")

    def to(self, device: torch.device | str) -> NamedAttribute:
        """Move all data and transforms to the specified device."""
        device = torch.device(device)  # Convert string to device if needed

        if hasattr(self, "scene_params") and self.scene_params is not None:
            self.scene_params = self.scene_params.to(device)
        if hasattr(self, "packed_data") and self.packed_data is not None:
            self.packed_data = self.packed_data.to(device)

        # Move all transforms that need device movement
        for transform in [self.coding, self.codebook, self.split]:
            if transform is not None:
                transform.to(device)
        return self

    def encode(self) -> Tensor:
        """Apply the encoding pipeline to scene_params to produce packed_data."""
        if not hasattr(self, "scene_params") or self.scene_params is None:
            raise ValueError("No scene_params available for encoding")

        data = self.scene_params
        for transform in self._transforms:
            data = transform.encode(data)

        self.packed_data = data
        return data

    def decode(self) -> Tensor:
        """Apply the decoding pipeline to packed_data to reconstruct scene_params."""

        if not hasattr(self, "packed_data") or self.packed_data is None:
            raise ValueError("No packed_data available for decoding")

        data = self.packed_data
        for transform in reversed(self._transforms):
            data = transform.decode(data)

        self.scene_params = data
        return data
