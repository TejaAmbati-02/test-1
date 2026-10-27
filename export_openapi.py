"""Write the API's OpenAPI 3.1.0 spec to openapi.json (or the path given).

Builds the spec from the FastAPI app itself, so it always matches the code. No
server, OpenSearch, Redis or OCI connection is needed — only the imports.

Usage:
    python export_openapi.py                # -> openapi.json
    python export_openapi.py docs/api.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app import app


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "openapi.json"
    spec = app.openapi()
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote OpenAPI {spec['openapi']} spec ({len(spec['paths'])} paths) to {out}")


if __name__ == "__main__":
    main()
