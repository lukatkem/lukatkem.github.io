"""mcpserver — a Model Context Protocol server built from scratch.

Hand-written JSON-RPC 2.0 over stdio, Python standard library only: no
official MCP SDK, no third-party dependencies, no network access.

Module map:

* ``mcpserver.protocol`` — JSON-RPC 2.0 dispatch, MCP method surface, stdio loop
* ``mcpserver.tools``    — the four text-toolkit tools and their JSON schemas
* ``mcpserver.server``   — executable entry point (``python -m mcpserver.server``)
* ``mcpserver.cli``      — ``python -m mcpserver`` wrapper (argparse + version flag)
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
