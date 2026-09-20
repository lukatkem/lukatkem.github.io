"""Command-line interface: ``python -m mcpserver``.

Behaviorally identical to ``python -m mcpserver.server``. The server speaks
newline-delimited JSON-RPC 2.0 on stdin/stdout, so it is normally launched
by an MCP client rather than used interactively.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .protocol import serve

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcpserver",
        description=(
            "Model Context Protocol (MCP) stdio server, implemented from "
            "scratch with the Python standard library. Reads newline-delimited "
            "JSON-RPC 2.0 requests on stdin and writes responses to stdout."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"mcpserver {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the stdio server loop until EOF."""
    build_parser().parse_args(argv)
    try:
        serve()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
