from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from models.response import ApiError
from services import image_processor as ip
from services.image_processor import MAX_IMAGE_BYTES, process_image

MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def _image_bytes(fmt: str, size=(40, 20), mode="RGB") -> bytes:
    out = BytesIO()
    Image.new(mode, size, "red").save(out, format=fmt)
    return out.getvalue()


def _part(data: bytes, content_type: str) -> UploadFile:
    return UploadFile(BytesIO(data), filename="photo", headers=Headers({"content-type": content_type}))


def _open(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def test_png_stays_png():
    data, mime = process_image(_part(_image_bytes("PNG"), "image/png"))
    assert mime == "image/png"
    assert _open(data).format == "PNG"


@pytest.mark.parametrize("fmt, mode", [("JPEG", "RGB"), ("WEBP", "RGBA")])
def test_other_formats_become_rgb_jpeg(fmt, mode):
    data, mime = process_image(_part(_image_bytes(fmt, mode=mode), MIME[fmt]))
    assert mime == "image/jpeg"
    img = _open(data)
    assert (img.format, img.mode) == ("JPEG", "RGB")


def test_large_images_are_scaled_down_keeping_aspect_ratio(monkeypatch):
    monkeypatch.setattr(ip.settings, "image_max_pixels", 200)
    data, _ = process_image(_part(_image_bytes("PNG", size=(40, 20)), "image/png"))  # 800 px -> scale 0.5
    assert _open(data).size == (20, 10)


def test_small_images_keep_their_size():
    data, _ = process_image(_part(_image_bytes("JPEG", size=(40, 20)), "image/jpeg"))
    assert _open(data).size == (40, 20)


def test_the_bytes_decide_the_format():
    data, mime = process_image(_part(_image_bytes("PNG"), "image/jpeg"))
    assert mime == "image/png"


@pytest.mark.parametrize("part", [None, "", _part(b"", "image/png")])
def test_no_image(part):
    assert process_image(part) is None


@pytest.mark.parametrize("part, status, code", [
    ("aGVsbG8=", 400, "UNSUPPORTED_IMAGE_TYPE"),                                   # Base64 text field
    (_part(_image_bytes("PNG"), "application/octet-stream"), 400, "UNSUPPORTED_IMAGE_TYPE"),
    (_part(_image_bytes("GIF"), "image/gif"), 400, "UNSUPPORTED_IMAGE_TYPE"),
    (_part(_image_bytes("GIF"), "image/png"), 400, "UNSUPPORTED_IMAGE_TYPE"),     # GIF bytes, PNG label
    (_part(b"\x89PNG not really a png", "image/png"), 400, "UNSUPPORTED_IMAGE_TYPE"),  # undecodable
    (_part(b"\x89PNG" + b"\x00" * MAX_IMAGE_BYTES, "image/png"), 413, "IMAGE_TOO_LARGE"),
])
def test_rejected(part, status, code):
    with pytest.raises(ApiError) as exc:
        process_image(part)
    assert exc.value.status_code == status
    assert exc.value.detail["code"] == code
