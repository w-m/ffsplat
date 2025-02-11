import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ..io.ply import load_ply, save_ply


@dataclass
class ConvertConfig:
    input_path: Annotated[Path, MISSING]
    output_path: Annotated[Path, MISSING]


cs = ConfigStore.instance()
cs.store(name="convert_config", node=ConvertConfig)


@hydra.main(version_base=None, config_path=None, config_name="convert_config")
def main(cfg: ConvertConfig) -> None:
    """Convert input PLY file to output PLY file."""
    # Convert string paths to Path objects if they aren't already
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {output_path.parent}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        gaussians = load_ply(input_path)
    except Exception as e:
        print(f"Error loading PLY file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        save_ply(gaussians, output_path)
    except Exception as e:
        print(f"Error saving PLY file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully converted {input_path} to {output_path}")


if __name__ == "__main__":
    main()
