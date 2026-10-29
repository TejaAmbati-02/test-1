"""Write the API's OpenAPI 3.1.0 spec to openapi.json (or the path given).

Also writes swagger.html next to it — a standalone Swagger UI page for the spec
(see swagger_html.py).

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
from swagger_html import write_swagger_html


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "openapi.json"
    spec = app.openapi()
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote OpenAPI {spec['openapi']} spec ({len(spec['paths'])} paths) to {out}")
    html = out.with_name("swagger.html")
    write_swagger_html(spec, html)
    print(f"Wrote Swagger UI to {html}")


if __name__ == "__main__":
    main()
