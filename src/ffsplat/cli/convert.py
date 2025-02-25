import sys
from pathlib import Path

from jsonargparse import ArgumentParser

from ..coding.scene_decoder import decode_gaussians
from ..coding.sog_arxiv import encode_sog_arxiv
from ..io.ply import save_ply


def main() -> None:
    """Convert between different scene formats.

    Supports conversion between PLY files and other formats, with configurable encoding options.
    """
    parser = ArgumentParser(
        description="Convert between different scene formats",
        default_config_files=["config.yaml"],
    )
    parser.add_argument("--input", type=Path, required=True, help="Input file or directory path")
    # TODO: add support for guessing input format
    parser.add_argument(
        "--input-format",
        type=str,
        required=True,
        help="Input format",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory path")
    # parser.add_argument("--codec", type=Path, help="Path to encoding YAML configuration file")

    cfg = parser.parse_args()

    if not cfg.input.exists():
        print(f"Error: Input path not found: {cfg.input}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {cfg.output.parent}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error creating output directory: {e}", file=sys.stderr)
        sys.exit(1)

    gaussians = decode_gaussians(input_path=cfg.input, input_format=cfg.input_format)

    try:
        save_ply(gaussians, cfg.output)
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)

    if False:
        try:
            encode_sog_arxiv(input_gaussians=gaussians, output_path=cfg.output)
        except Exception as e:
            print(f"Error encoding file: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"Successfully converted {cfg.input} to {cfg.output}")


if __name__ == "__main__":
    main()
