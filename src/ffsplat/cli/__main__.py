"""ffsplat command-line entry point.

This module provides a single ``ffsplat`` executable that in turn dispatches
to the individual sub-command implementations that already live in
``ffsplat.cli.*`` (``convert``, ``view``, ``live``, ``eval`` …).

We refrain from recreating the full argument definitions here because the
sub-modules already expose a fully-featured ``main()`` function that sets up
their own argument parser.  Re-implementing those arguments in this file
would lead to duplication that is hard to keep in sync and easy to break.

Instead we implement a *minimal* dispatcher: the first positional argument
determines which sub-command should be executed.  Everything that follows is
forwarded unchanged to the respective ``main()`` function by manipulating
``sys.argv`` before the call - exactly how Python would have invoked the
sub-command if it had been installed as its own console script.

Examples
--------
>>> ffsplat convert --input scene.ply --input-format ply \
                   --output scene.sog --output-format SOG-web

>>> ffsplat view --input scene.sog --input-format SOG-web
"""

from __future__ import annotations

import importlib.metadata as _metadata
import sys
from textwrap import dedent


def _usage() -> str:
    """Return a short usage string for the root command."""

    return dedent(
        """
        usage: ffsplat <command> [...]

        Commands
        --------
          convert   Convert between different scene formats.
          view      View or inspect a scene interactively.
          live      Real-time conversion and visualisation pipeline.
          eval      Evaluate rate-distortion performance.

        Use "ffsplat <command> --help" for additional information.
        """
    ).strip()


def main() -> None:
    """Entry point that dispatches to *ffsplat* sub-commands."""

    subcommand_mapping = {
        "convert": "ffsplat.cli.convert",
        "view": "ffsplat.cli.view",
        "live": "ffsplat.cli.live",
        "eval": "ffsplat.cli.eval",
    }

    if len(sys.argv) == 1:
        # Called without any arguments - show usage and exit.
        print(_usage())
        sys.exit(0)

    # Handle *root-level* flags that should be intercepted before dispatch.
    if sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        sys.exit(0)

    if sys.argv[1] in {"-V", "--version"}:
        try:
            version = _metadata.version("ffsplat")
        except _metadata.PackageNotFoundError:
            version = "unknown"
        print(f"ffsplat {version}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in subcommand_mapping:
        print(f"ffsplat: error: unknown command '{cmd}'", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        sys.exit(1)

    # Re-write *sys.argv* so the sub-command sees the expected argv[0].
    # Example: ["ffsplat", "convert", "--input", ...] →
    #          ["ffsplat convert", "--input", ...]
    sys.argv = [f"ffsplat {cmd}"] + sys.argv[2:]

    # Import lazily so that heavy dependencies of unrelated commands are not
    # loaded unless necessary.
    module_name = subcommand_mapping[cmd]
    module = __import__(module_name, fromlist=["main"])

    if not hasattr(module, "main"):
        # This should never happen but fail gracefully.
        print(f"ffsplat: internal error: '{module_name}' has no 'main' function", file=sys.stderr)
        sys.exit(1)

    # Delegate execution.
    module.main()  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover - executed when run as a module
    main()
