from dataclasses import dataclass

import torch
from imagecodecs import jpegxl_decode, jpegxl_encode
from torch import Tensor

from ..models.encoding_transform import EncodingTransform


@dataclass
class JPEGXLEncodingConfig:
    level: int


class JPEGXLCoding(EncodingTransform[JPEGXLEncodingConfig, None]):
    def _encode_impl(self, data: Tensor, config: JPEGXLEncodingConfig) -> Tensor:
        numpy_data = data.cpu().numpy()
        buf: bytes = jpegxl_encode(numpy_data, level=config.level)
        return torch.frombuffer(buf, dtype=torch.uint8).to(data.device)

    def _decode_impl(self, data: Tensor, params: None) -> Tensor:
        buf = bytes(data.cpu().numpy().tobytes())
        decoded = jpegxl_decode(buf)
        return torch.from_numpy(decoded).to(data.device)
