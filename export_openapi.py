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
from utils.logger import logger


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "openapi.json"
    logger.debug(f"[export_openapi.main] start — out={out}")
    try:
        spec = app.openapi()
        logger.debug(f"[export_openapi.main] generated OpenAPI {spec.get('openapi')} spec with {len(spec.get('paths', {}))} paths")
        out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote OpenAPI {spec['openapi']} spec ({len(spec['paths'])} paths) to {out}")
        html = out.with_name("swagger.html")
        write_swagger_html(spec, html)
        print(f"Wrote Swagger UI to {html}")
        logger.info(f"[export_openapi.main] wrote {out} and {html}")
    except Exception as exc:
        logger.exception(
            f"[export_openapi.main] failed to export OpenAPI spec to {out} ({type(exc).__name__}): {exc}",
            extra={"function_name": "export_openapi.main", "status": "error", "error_details": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
