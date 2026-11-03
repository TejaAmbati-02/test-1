"""Write a standalone Swagger UI page (swagger.html) for openapi.json.

The spec is embedded in the page, so it opens straight from disk (file://) with
no running server — useful where OpenSearch, Redis or OCI are unreachable and
the app's own /docs cannot start. Swagger UI itself loads from the jsDelivr CDN.

"Try it out" sends requests to the server given with --server (added to the
spec's ``servers``); without it the page is read-only documentation.

Usage:
    python swagger_html.py                                   # openapi.json -> swagger.html
    python swagger_html.py --server https://<api host>       # enable "Try it out"
    python swagger_html.py docs/api.json docs/swagger.html
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

# Standalone script: plain stdlib logging so it runs without the app's .env/settings.
logger = logging.getLogger("swagger_html")

HERE = Path(__file__).parent

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title} — Swagger UI</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({{
      spec: {spec},
      dom_id: "#swagger-ui",
      deepLinking: true,
      persistAuthorization: true,
    }});
  </script>
</body>
</html>
"""


def write_swagger_html(spec: dict, out: Path, server: str | None = None) -> None:
    """Write a Swagger UI page with ``spec`` embedded to ``out``."""
    logger.debug(f"[write_swagger_html] start — out={out} server={server or '-'} paths={len(spec.get('paths', {}))}")
    try:
        if server:
            spec = {**spec, "servers": [{"url": server.rstrip("/")}]}
        # "</" would end the inline <script> early if it appears in any description.
        embedded = json.dumps(spec, ensure_ascii=False).replace("</", "<\\/")
        title = spec.get("info", {}).get("title", "API")
        out.write_text(_TEMPLATE.format(title=title, spec=embedded), encoding="utf-8")
        logger.info(f"[write_swagger_html] wrote '{title}' Swagger UI to {out} ({len(embedded)} chars of spec)")
    except Exception as exc:
        logger.exception(
            f"[write_swagger_html] failed to write {out} ({type(exc).__name__}): {exc}",
            extra={"function_name": "write_swagger_html", "status": "error", "error_details": str(exc)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="?", type=Path, default=HERE / "openapi.json")
    parser.add_argument("out", nargs="?", type=Path, default=HERE / "swagger.html")
    parser.add_argument("--server", help="Base URL the 'Try it out' requests go to")
    args = parser.parse_args()
    try:
        logger.debug(f"[main] reading spec from {args.spec}")
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        write_swagger_html(spec, args.out, args.server)
        print(f"Wrote Swagger UI for {args.spec.name} to {args.out}")
    except FileNotFoundError as exc:
        logger.error(f"[main] spec file not found: {args.spec} — {exc}",
                     extra={"function_name": "main", "status": "error", "error_details": str(exc)})
        raise
    except json.JSONDecodeError as exc:
        logger.error(f"[main] {args.spec} is not valid JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}",
                     extra={"function_name": "main", "status": "error", "error_details": str(exc)})
        raise
    except Exception as exc:
        logger.exception(f"[main] failed ({type(exc).__name__}): {exc}",
                         extra={"function_name": "main", "status": "error", "error_details": str(exc)})
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    main()
