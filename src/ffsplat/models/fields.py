from dataclasses import dataclass

from torch import Tensor


@dataclass
class NamedField:
    name: str
    data: Tensor
