"""Root entrypoint for Knowledge Brain MCP server."""

import os
import sys

# Ensure knowledge-brain directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server.mcp_server import (
    create_mcp_server,
    mcp,
    main,
)

__all__ = [
    "create_mcp_server",
    "mcp",
    "main",
]

if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run()

