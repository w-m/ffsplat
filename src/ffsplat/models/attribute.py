from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Literal, overload

import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch import device as torch_device
from torch import dtype as torch_dtype

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
    device_str: str
    data: Tensor
    to_device_fn: Callable[[Tensor, str], Tensor]

    def __init__(
        self,
        name: str,
        data: Tensor,
        device: str = "cuda",
        to_device_fn: Callable[[Tensor, str], Tensor] | None = None,
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
        self.device_str = device  # Store string version separately
        self.to_device_fn = to_device_fn or (lambda x, d: x.to(d))

        # Register the data as a buffer so it moves with the module
        self.register_buffer("data", self.to_device_fn(data, device))

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
            self.flatten = FlattenTransform(flatten_start_dim, flatten_end_dim)

    @property
    def device(self) -> str:
        """Get the current device as a string."""
        return self.device_str

    @overload
    def to(
        self,
        device: str | torch_device | int | None,
        *,
        dtype: torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> NamedAttribute: ...

    @overload
    def to(self, dtype_arg: torch_dtype, *, non_blocking: bool = False) -> NamedAttribute: ...

    @overload
    def to(self, tensor: Tensor, *, non_blocking: bool = False) -> NamedAttribute: ...

    def to(
        self,
        device_or_dtype: str | torch_device | torch_dtype | Tensor | int | None,
        *,
        dtype: torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> NamedAttribute:
        """Move the attribute to the specified device/dtype."""
        if isinstance(device_or_dtype, (str, torch_device, int)) or device_or_dtype is None:
            # Handle device move
            device_str = str(device_or_dtype) if device_or_dtype is not None else self.device_str
            if device_str != self.device_str:
                self.data = self.to_device_fn(self.data, device_str)
                self.device_str = device_str
                super().to(device_str)
        elif isinstance(device_or_dtype, torch_dtype):
            # Handle dtype passed as first argument
            self.data = self.data.to(dtype=device_or_dtype, non_blocking=non_blocking)
        elif isinstance(device_or_dtype, Tensor):
            # Handle tensor device/dtype move
            self.data = self.data.to(device_or_dtype.device, dtype=device_or_dtype.dtype, non_blocking=non_blocking)

        # Handle explicit dtype parameter
        if dtype is not None:
            self.data = self.data.to(dtype=dtype, non_blocking=non_blocking)

        return self

    def forward(self, context: Mapping[str, Tensor | NamedAttribute] | None = None) -> Tensor:
        """Apply transforms in fixed order."""
        data = self.data

        # Convert context values from NamedAttribute to Tensor if needed
        if context is not None:
            context = {name: attr.data if isinstance(attr, NamedAttribute) else attr for name, attr in context.items()}

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


class BitshiftTransform(nn.Module):
    """Shift integer data left by shift_amount in forward; right for backward."""

    def __init__(self, shift_amount: int = 0) -> None:
        super().__init__()
        self.shift_amount = shift_amount

    def forward(self, data: Tensor) -> Tensor:
        if self.shift_amount != 0 and not torch.is_floating_point(data):
            return data << self.shift_amount
        return data

    def inverse(self, data: Tensor) -> Tensor:
        """Inverse transform - shift right instead of left."""
        if self.shift_amount != 0 and not torch.is_floating_point(data):
            return data >> self.shift_amount
        return data


class CombineTransform(nn.Module):
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


class DTypeTransform(nn.Module):
    def __init__(self, target_dtype: torch.dtype) -> None:
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, data: Tensor) -> Tensor:
        return data.to(self.target_dtype)

    def inverse(self, data: Tensor) -> Tensor:
        # For a perfect inverse, you'd have to know the original dtype.
        # We'll just demonstrate the concept here:
        return data


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


class FlattenTransform(nn.Module):
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
