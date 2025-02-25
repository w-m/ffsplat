from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "N 3"]
    quaternions: Float[Tensor, "N 4"]
    scales: Float[Tensor, "N 3"]
    opacities: Float[Tensor, " N"]
    sh: Float[Tensor, "N S 3"]
