"""Package entry point so ``python -m mcpserver`` works.

Delegates to :func:`mcpserver.cli.main`, which is behaviorally identical to
``python -m mcpserver.server``.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
