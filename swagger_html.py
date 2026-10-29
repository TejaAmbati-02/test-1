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
from pathlib import Path

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
    if server:
        spec = {**spec, "servers": [{"url": server.rstrip("/")}]}
    # "</" would end the inline <script> early if it appears in any description.
    embedded = json.dumps(spec, ensure_ascii=False).replace("</", "<\\/")
    title = spec.get("info", {}).get("title", "API")
    out.write_text(_TEMPLATE.format(title=title, spec=embedded), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="?", type=Path, default=HERE / "openapi.json")
    parser.add_argument("out", nargs="?", type=Path, default=HERE / "swagger.html")
    parser.add_argument("--server", help="Base URL the 'Try it out' requests go to")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    write_swagger_html(spec, args.out, args.server)
    print(f"Wrote Swagger UI for {args.spec.name} to {args.out}")


if __name__ == "__main__":
    main()
