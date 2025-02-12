from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal

import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .exceptions import (
    BackwardNotImplementedError,
    CodebookConfigError,
    InsufficientDimensionsError,
    MissingContextError,
    ShapeMismatchError,
    TrailingDimensionMismatchError,
    UnknownCombineMethodError,
    UnknownRemappingMethodError,
    ZeroRangeError,
)


class EncodingTransform(nn.Module):
    def _init_from_encoding_config(self, **encoding_config) -> None:
        self._encoding_config = encoding_config
        self._decoding_params = {}

    def _init_from_decoding_params(self, **decoding_params) -> None:
        self._encoding_config = {}
        self._decoding_params = decoding_params

    def encode(self, data: Tensor) -> Tensor:
        return data

    def decode(self, data: Tensor) -> Tensor:
        return data

    @property
    def encoding_config(self) -> dict[str, Any]:
        return self._encoding_config

    @property
    def decoding_params(self) -> dict[str, Any]:
        return self._decoding_params

    def forward(self, data: Tensor) -> Tensor:
        """Forward processing is decoding raw data into a form suitable for rendering."""
        return self.decode(data)

    def inverse(self, data: Tensor) -> Tensor:
        """Backward processing is encoding data into a form suitable for storage."""
        return self.encode(data)

    @classmethod
    def from_encoding_config(cls, **encoding_config) -> EncodingTransform:
        """Create an encoding transform from encoding configuration."""
        obj = cls()
        obj._init_from_encoding_config(**encoding_config)
        return obj

    @classmethod
    def from_decoding_params(cls, **decoding_params) -> EncodingTransform:
        """Create an encoding transform from decoding parameters."""
        obj = cls()
        obj._init_from_decoding_params(**decoding_params)
        return obj


class QuantizationTransform(EncodingTransform):
    """Quantize the input tensor to a fixed number of bits."""

    def _init_from_encoding_config(self, num_bits: int) -> None:
        super()._init_from_encoding_config(num_bits=num_bits)

        if self.num_bits <= 0:
            raise ValueError(f"Number of bits in quantization must be a positive integer, got {num_bits}")

        self.num_bits = num_bits

    def encode(self, data: Tensor) -> Tensor:
        # Quantize the data to the specified number of bits
        # Note: We're assuming the input is in the range [0, 1]
        # For a more general case, you'd need to scale the data to [0, 1] first
        return torch.round(data * (2**self.num_bits - 1)) / (2**self.num_bits - 1)


class TrimmingTransform(EncodingTransform):
    """Trim away a number of elements by removing the last N elements."""

    def _init_from_encoding_config(self, num_elements_to_trim: int) -> None:
        super()._init_from_encoding_config(num_elements_to_trim=num_elements_to_trim)
        self.num_elements_to_trim = num_elements_to_trim

    def encode(self, data: Tensor) -> Tensor:
        return data[: -self.num_elements_to_trim]


class ClampingTransform(EncodingTransform):
    """Clamp the input tensor to a specified range."""

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__(min_val=min_val, max_val=max_val)
        self.min_val = min_val
        self.max_val = max_val

    def encode(self, data: Tensor) -> Tensor:
        return torch.clamp(data, self.min_val, self.max_val)


class NamedAttribute(nn.Module):
    """
    An attribute with a name and data tensor, with optional transforms applied in a fixed order:
    1. Initial dtype conversion + bitshift (optional)
    2. Combine data with other attributes (optional)
    3. Secondary dtype conversion (optional)
    4. Rescaling (optional)
    5. Remapping (optional)
    6. Flatten (optional)
    """

    name: str

    raw_data: Tensor
    coding: EncodingTransform
    reshaping: GridTransform
    trimming: TrimmingTransform
    coding_dtype: DTypeTransform
    quantization: QuantizationTransform
    bit_shift: BitshiftTransform
    combine: CombineTransform
    combined_dtype: DTypeTransform
    rescaling: RescalingTransform
    clamping: ClampingTransform
    remapping: RemappingTransform

    def __init__(
        self,
        # force keyword arguments for clarity
        *,
        name: str,
        data: Tensor,
        # Transform params with defaults for no-op behavior
        initial_dtype: torch.dtype | None = None,
        bitshift_amount: int = 0,
        combine_method: Literal["concat", "logical_or", "codebook_lookup"] | None = None,
        combine_attrs: list[str] | None = None,
        combine_axis: int = 1,
        codebook_attr: str | None = None,
        index_attr: str | None = None,
        target_dtype: torch.dtype | None = None,
        rescale_min: Tensor | None = None,
        rescale_max: Tensor | None = None,
        remap_method: Literal["exp", "sigmoid"] | None = None,
        flatten_start_dim: int | None = None,
        flatten_end_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.name = name

        # Register the data as a buffer so it moves with the module
        self.register_buffer("data", data)

        # Initialize transforms that we'll use (all default to no-op)
        if initial_dtype is not None:
            self.initial_dtype_transform = DTypeTransform(initial_dtype)

        if bitshift_amount != 0:
            self.bitshift = BitshiftTransform(bitshift_amount)

        if combine_method is not None:
            self.combine = CombineTransform(
                method=combine_method,
                other_attr_names=combine_attrs or [],
                axis=combine_axis,
                codebook_attr=codebook_attr,
                index_attr=index_attr,
            )

        if target_dtype is not None:
            self.target_dtype_transform = DTypeTransform(target_dtype)

        if rescale_min is not None and rescale_max is not None:
            self.rescale = RescalingTransform(rescale_min, rescale_max)

        if remap_method is not None:
            self.remap = RemappingTransform(remap_method)

        if flatten_start_dim is not None and flatten_end_dim is not None:
            self.flatten = GridTransform(flatten_start_dim, flatten_end_dim)

    @property
    def device(self) -> torch.device:
        return self.raw_data.device

    def to(self, device) -> NamedAttribute:
        """Move the data tensor to a new device."""
        if self.device == device:
            return self

    def forward(self, context: Mapping[str, Tensor | NamedAttribute] | None = None) -> Tensor:
        """Apply transforms in fixed order."""
        data = self.raw_data

        # Convert context values from NamedAttribute to Tensor if needed
        if context is not None:
            context = {
                name: attr.raw_data if isinstance(attr, NamedAttribute) else attr for name, attr in context.items()
            }

        # Step 1: Initial dtype + bitshift
        if hasattr(self, "initial_dtype_transform"):
            data = self.initial_dtype_transform(data)
        if hasattr(self, "bitshift"):
            data = self.bitshift(data)

        # Step 2: Combine (needs context)
        if hasattr(self, "combine"):
            if context is None:
                raise MissingContextError()
            data = self.combine(data, context)

        # Step 3: Secondary dtype
        if hasattr(self, "target_dtype_transform"):
            data = self.target_dtype_transform(data)

        # Step 4: Rescaling
        if hasattr(self, "rescale"):
            data = self.rescale(data)

        # Step 5: Remapping
        if hasattr(self, "remap"):
            data = self.remap(data)

        # Step 6: Flatten
        if hasattr(self, "flatten"):
            data = self.flatten(data)

        return data

    def __call__(self, context: Mapping[str, Tensor | NamedAttribute] | None = None) -> Tensor:
        """Apply transforms by calling forward()."""
        return self.forward(context)

    def inverse(self, data: Tensor) -> Tensor:
        """Apply transforms in reverse order."""
        # Step 6: Flatten (inverse)
        if hasattr(self, "flatten"):
            data = self.flatten.inverse(data)

        # Step 5: Remapping (inverse)
        if hasattr(self, "remap"):
            data = self.remap.inverse(data)

        # Step 4: Rescaling (inverse)
        if hasattr(self, "rescale"):
            data = self.rescale.inverse(data)

        # Step 3: Secondary dtype (inverse)
        if hasattr(self, "target_dtype_transform"):
            data = self.target_dtype_transform.inverse(data)

        # Step 2: Combine (inverse)
        if hasattr(self, "combine"):
            data = self.combine.inverse(data)

        # Step 1: Bitshift + Initial dtype (inverse)
        if hasattr(self, "bitshift"):
            data = self.bitshift.inverse(data)
        if hasattr(self, "initial_dtype_transform"):
            data = self.initial_dtype_transform.inverse(data)

        return data

    @classmethod
    def from_numpy(
        cls,
        name: str,
        data: NDArray[Any],
        device: str = "cuda",
        to_device_fn: Callable[[Tensor, str], Tensor] | None = None,
        **transform_kwargs: Any,
    ) -> NamedAttribute:
        """Create a NamedAttribute from a numpy array, converting to torch tensor"""
        tensor = torch.from_numpy(data)
        return cls(name=name, data=tensor, device=device, to_device_fn=to_device_fn, **transform_kwargs)


class BitshiftTransform(EncodingTransform):
    """Shift integer right for encoding, left for decoding"""

    def __init__(self, shift_amount: int = 0) -> None:
        super().__init__(shift_amount=shift_amount)

        if shift_amount < 0:
            raise ValueError(f"Shift amount must be non-negative, got {shift_amount}")

        self.shift_amount = shift_amount

    def encode(self, data: Tensor) -> Tensor:
        if torch.is_floating_point(data):
            raise ValueError("Bitshift transform is only for integer data")

        return data >> self.shift_amount

    def decode(self, data: Tensor) -> Tensor:
        if torch.is_floating_point(data):
            raise ValueError("Bitshift transform is only for integer data")

        return data << self.shift_amount

    def decoding_params(self) -> dict[str, Any]:
        return {"shift_amount": self.shift_amount}


class CombineTransform(EncodingTransform):
    """
    method='concat':  data = torch.concat([data, *others], dim=axis)
    method='logical_or':  data = reduce(operator.or_, (others), data)
    method='codebook_lookup': data = codebook[indices]
    """

    def __init__(
        self,
        method: Literal["concat", "logical_or", "codebook_lookup"],
        other_attr_names: list[str],
        axis: int = 1,
        codebook_attr: str | None = None,
        index_attr: str | None = None,
    ):
        super().__init__()
        self.method = method
        self.other_attr_names = other_attr_names
        self.axis = axis
        self.codebook_attr = codebook_attr
        self.index_attr = index_attr

    def forward(self, data: Tensor, context_tensors: dict[str, Tensor]) -> Tensor:
        if self.method == "concat":
            others = [context_tensors[name] for name in self.other_attr_names]
            return torch.concat([data, *others], dim=self.axis)

        elif self.method == "logical_or":
            import functools
            import operator

            return functools.reduce(operator.or_, (context_tensors[n] for n in self.other_attr_names), data)

        elif self.method == "codebook_lookup":
            if not self.codebook_attr or not self.index_attr:
                raise CodebookConfigError()
            codebook = context_tensors[self.codebook_attr]
            indices = context_tensors[self.index_attr]
            return codebook[indices]
        else:
            raise UnknownCombineMethodError(self.method)

    def inverse(self, data: Tensor) -> Tensor:
        """Inverse transform operation."""
        raise BackwardNotImplementedError(self.method)


class DTypeTransform(EncodingTransform):
    def __init__(self, source_dtype: torch.dtype, target_dtype: torch.dtype) -> None:
        super().__init__(source_dtype=source_dtype, target_dtype=target_dtype)
        self.source_dtype = source_dtype
        self.target_dtype = target_dtype

    def encode(self, data: Tensor) -> Tensor:
        if data.dtype != self.source_dtype:
            raise ValueError(f"Expected data to have dtype {self.source_dtype}, got {data.dtype}")
        return data.to(self.target_dtype)

    def decode(self, data: Tensor) -> Tensor:
        if data.dtype != self.target_dtype:
            raise ValueError(f"Expected data to have dtype {self.target_dtype}, got {data.dtype}")
        return data.to(self.source_dtype)


class RescalingTransform(nn.Module):
    """
    The forward pass does: data = (data - min) / (max - min).
    The backward pass does: data = data * (max - min) + min.

    min_val and max_val are tensors that are applied to the last N dimensions of data,
    where N is the number of dimensions in min_val/max_val. The data tensor can have
    additional dimensions in front.

    Example:
    data shape: (1000, 32, 3)  # 1000 points, 32 harmonics, 3 channels
    min_val shape: (32, 3)     # per-harmonic, per-channel min values
    max_val shape: (32, 3)     # per-harmonic, per-channel max values
    """

    min_val: Tensor
    max_val: Tensor

    def __init__(self, min_val: Tensor, max_val: Tensor) -> None:
        super().__init__()
        if min_val.shape != max_val.shape:
            raise ShapeMismatchError(tuple(min_val.shape), tuple(max_val.shape))
        # Store tensors as buffers so they move to the right device with the module
        self.register_buffer("min_val", min_val)
        self.register_buffer("max_val", max_val)

    def _validate_shapes(self, data: Tensor) -> None:
        """Check that the data's trailing dimensions match min/max dimensions."""
        # Note: self.min_val is guaranteed to be a Tensor due to class attribute annotation
        min_shape = tuple(self.min_val.shape)
        min_dims = len(min_shape)
        data_dims = len(data.shape)

        if data_dims < min_dims:
            raise InsufficientDimensionsError(data_dims, min_dims, min_shape)

        if tuple(data.shape[-min_dims:]) != min_shape:
            raise TrailingDimensionMismatchError(tuple(data.shape[-min_dims:]), min_shape)

    def forward(self, data: Tensor) -> Tensor:
        self._validate_shapes(data)
        # Get tensors and ensure they match data's device and dtype
        # Note: self.min_val and self.max_val are guaranteed to be Tensors
        min_val = self.min_val.to(device=data.device, dtype=data.dtype)
        max_val = self.max_val.to(device=data.device, dtype=data.dtype)
        diff = max_val - min_val
        if torch.any(diff == 0):
            raise ZeroRangeError()
        return (data - min_val) / diff

    def inverse(self, data: Tensor) -> Tensor:
        """Inverse transform: x' = x * (max - min) + min"""
        self._validate_shapes(data)
        # Get tensors and ensure they match data's device and dtype
        min_val = self.min_val.to(device=data.device, dtype=data.dtype)
        max_val = self.max_val.to(device=data.device, dtype=data.dtype)
        diff = max_val - min_val
        return data * diff + min_val


class RemappingTransform(nn.Module):
    """
    method='exp' => forward: data = exp(data), backward: data = log(data)
    method='sigmoid' => forward: data = sigmoid(data), backward: data = logit(data)
    """

    def __init__(self, method: Literal["exp", "sigmoid"]) -> None:
        super().__init__()
        self.method = method

    def forward(self, data: Tensor) -> Tensor:
        if self.method == "exp":
            return torch.exp(data)
        elif self.method == "sigmoid":
            return torch.sigmoid(data)
        else:
            raise UnknownRemappingMethodError(self.method)

    def inverse(self, data: Tensor) -> Tensor:
        """Inverse transform operation."""
        if self.method == "exp":
            # Inverse of exp => log
            # Watch out for zero values, etc.
            return torch.log(data)
        elif self.method == "sigmoid":
            # Inverse of sigmoid => logit
            eps = 1e-7
            return torch.log(data.clamp(eps, 1 - eps) / (1 - data.clamp(eps, 1 - eps)))
        else:
            return data


class GridTransform(nn.Module):
    def __init__(self, start_dim: int, end_dim: int) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, data: Tensor) -> Tensor:
        # Use torch.flatten instead of F.flatten since F.flatten returns Any
        return torch.flatten(data, self.start_dim, self.end_dim)

    def inverse(self, data: Tensor) -> Tensor:
        """Inverse transform operation."""
        raise BackwardNotImplementedError("flatten")
