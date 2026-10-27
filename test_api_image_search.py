"""Known-answer image search tests against the running API (POST /v1/search).

Each image in ``test_images_known/`` is a byte-identical copy of a product image
and ``expected.json`` names the SKU(s) it belongs to, so the first product
returned must be one of those SKUs. Also checks that ``request_id``
reuse returns the same top hit, and that the upload rules hold: the image is
binary in a multipart file part with Content-Type image/jpeg, image/png or
image/webp (max 10 MiB); a text field (Base64), any other Content-Type, GIF or
non-image bytes get 400 and larger files 413.

Start the API first (python app.py), then:
    python test_api_image_search.py
    python test_api_image_search.py --base-url http://localhost:8000 --only ao-100018
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).parent


def post_search(
    base_url: str, fields: dict[str, str], image: Path | None = None, content_type: str | None = None,
) -> tuple[int, dict]:
    """POST a multipart /v1/search request; return (status, JSON body).

    The image goes out as a normal image part with the Content-Type guessed from
    its file name, unless ``content_type`` overrides it.
    """
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    if image is not None:
        # A normal image part, as the frontend sends it; the API detects the format from the bytes.
        ctype = content_type or mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{image.name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode() + image.read_bytes() + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{base_url}/v1/search", data=b"".join(parts), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, {"raw": body[:200].decode(errors="replace")}


BASE_FIELDS = {"channel": "web", "language": "en", "page": "1", "page_size": "10"}


def run(base_url: str, cases_dir: Path, bad_dir: Path, only: str | None) -> int:
    cases = json.loads((cases_dir / "expected.json").read_text())["cases"]
    if only:
        cases = [c for c in cases if only in c["image"]]
    passed = failed = 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal passed, failed
        passed += ok
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<42} {detail}")

    print(f"\n  Image search — {base_url}/v1/search\n")
    for c in cases:
        status, body = post_search(base_url, BASE_FIELDS, cases_dir / c["image"])
        products = body.get("products") or []
        top = products[0].get("sku_id") if products else None
        ok = status == 200 and top in c["expected_skus"]
        report(c["image"], ok, f"status={status} top={top} total={body.get('total_results')}"
               + ("" if ok else f" expected={c['expected_skus'][:3]} body={str(body)[:120]}"))

        # Reuse the cached embedding: same top hit without re-uploading the image.
        req_id = body.get("request_id")
        if ok and req_id:
            status2, body2 = post_search(base_url, {**BASE_FIELDS, "request_id": req_id})
            top2 = ((body2.get("products") or [{}])[0]).get("sku_id")
            report(f"  refine by request_id", status2 == 200 and top2 == top and body2.get("request_id") == req_id,
                   f"status={status2} top={top2}")

    # Upload rules: GIF or non-image bytes -> 400, an image over 10 MiB -> 413.
    for name, code in (("sample.gif", 400), ("not_an_image.png", 400), ("oversized.jpg", 413)):
        path = bad_dir / name
        if path.exists():
            status, body = post_search(base_url, BASE_FIELDS, path)
            report(name, status == code, f"status={status} (want {code}) body={str(body)[:100]}")

    # Only a normal image part is accepted: a JPEG sent as application/octet-stream,
    # or the image sent as a Base64 text field, -> 400.
    jpg = bad_dir / "ao-100018_1.jpg"
    if jpg.exists():
        status, body = post_search(base_url, BASE_FIELDS, jpg, content_type="application/octet-stream")
        report("jpg sent as application/octet-stream", status == 400, f"status={status} (want 400) body={str(body)[:100]}")
        b64 = base64.b64encode(jpg.read_bytes()).decode()
        status, body = post_search(base_url, {**BASE_FIELDS, "image": b64})
        report("image sent as Base64 text field", status == 400, f"status={status} (want 400) body={str(body)[:100]}")

    print(f"\n  {passed}/{passed + failed} passed\n")
    return 0 if failed == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Known-answer image search tests against the API.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--cases-dir", type=Path, default=HERE / "test_images_known")
    p.add_argument("--bad-dir", type=Path, default=HERE / "test_images", help="Folder with sample.gif / not_an_image.png / oversized.jpg / ao-100018_1.jpg")
    p.add_argument("--only", help="Run only cases whose image file name contains this text.")
    a = p.parse_args()
    sys.exit(run(a.base_url.rstrip("/"), a.cases_dir, a.bad_dir, a.only))


if __name__ == "__main__":
    main()
