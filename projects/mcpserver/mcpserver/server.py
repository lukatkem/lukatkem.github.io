"""Executable entry point: the MCP stdio server loop.

Run directly::

    python -m mcpserver.server

or equivalently via the CLI wrapper::

    python -m mcpserver
"""

from __future__ import annotations

import sys

from .protocol import serve

__all__ = ["main"]


def main() -> int:
    """Run the stdio server until EOF; return a process exit code."""
    try:
        serve()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
