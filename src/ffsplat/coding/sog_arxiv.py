from pathlib import Path

from ..models.gaussians import Gaussians


def encode_sog_arxiv(input_gaussians: Gaussians, output_path: Path) -> None:
    # produce scene_params
    input_gaussians.decode()

    # config = GaussiansEncodingConfig(
    #     means=

    # input_gaussians.means_attr.coding = )

    # input_gaussians.encode()

    pass
