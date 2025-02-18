from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Self, TypeVar

if TYPE_CHECKING:
    pass

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
