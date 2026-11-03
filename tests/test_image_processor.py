"""Tests for services/image_processor.py (EXIF fix, downscale, re-encode)."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from services.image_processor import ImageProcessor, InvalidImageError, image_processor


def _encode(fmt: str, size=(10, 10), mode="RGB", exif_orientation: int | None = None) -> bytes:
    image = Image.new(mode, size, color=0)
    out = BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


def _open(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


@pytest.fixture
def processor() -> ImageProcessor:
    return ImageProcessor(max_pixels=10_000, jpeg_quality=85)


def test_from_settings_uses_config_values():
    cfg = SimpleNamespace(image_max_pixels=123, image_jpeg_quality=77)
    p = ImageProcessor.from_settings(cfg)
    assert (p.max_pixels, p.jpeg_quality) == (123, 77)


def test_from_settings_propagates_errors():
    with pytest.raises(AttributeError):
        ImageProcessor.from_settings(SimpleNamespace())


def test_module_instance_is_built():
    assert isinstance(image_processor, ImageProcessor)


def test_png_stays_png(processor):
    data, mime = processor.process(_encode("PNG", mode="RGBA"))
    assert mime == "image/png"
    img = _open(data)
    assert img.format == "PNG"
    assert img.mode == "RGBA"


def test_jpeg_stays_jpeg(processor):
    data, mime = processor.process(_encode("JPEG"))
    assert mime == "image/jpeg"
    assert _open(data).format == "JPEG"


@pytest.mark.parametrize("fmt,mode", [("GIF", "P"), ("WEBP", "RGBA"), ("BMP", "L")])
def test_other_formats_become_rgb_jpeg(processor, fmt, mode):
    data, mime = processor.process(_encode(fmt, mode=mode))
    assert mime == "image/jpeg"
    img = _open(data)
    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_small_image_is_not_resized(processor):
    data, _ = processor.process(_encode("PNG", size=(50, 40)))
    assert _open(data).size == (50, 40)


def test_large_image_is_downscaled_keeping_aspect_ratio(processor):
    data, _ = processor.process(_encode("PNG", size=(400, 100)))
    w, h = _open(data).size
    assert w * h <= processor.max_pixels
    assert (w, h) == (200, 50)


def test_extreme_aspect_ratio_keeps_min_dimension_of_one():
    p = ImageProcessor(max_pixels=4, jpeg_quality=85)
    data, _ = p.process(_encode("PNG", size=(1000, 1)))
    w, h = _open(data).size
    assert h == 1 and w >= 1


def test_exif_orientation_is_applied(processor):
    # Orientation 6 = rotate 90° clockwise on display, so width/height swap.
    data, _ = processor.process(_encode("JPEG", size=(40, 20), exif_orientation=6))
    assert _open(data).size == (20, 40)


def test_jpeg_quality_affects_output_size():
    src = Image.effect_noise((80, 80), 64).convert("RGB")
    raw = BytesIO()
    src.save(raw, format="BMP")
    low, _ = ImageProcessor(max_pixels=1_000_000, jpeg_quality=10).process(raw.getvalue())
    high, _ = ImageProcessor(max_pixels=1_000_000, jpeg_quality=95).process(raw.getvalue())
    assert len(low) < len(high)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32])
def test_undecodable_bytes_raise_invalid_image(processor, data):
    with pytest.raises(InvalidImageError):
        processor.process(data)


def test_invalid_image_error_is_value_error():
    assert issubclass(InvalidImageError, ValueError)


def test_decompression_bomb_raises_invalid_image(processor, monkeypatch):
    def boom(*_a, **_k):
        raise Image.DecompressionBombError("too big")

    monkeypatch.setattr("services.image_processor.Image.open", boom)
    with pytest.raises(InvalidImageError, match="too big"):
        processor.process(b"x")


def test_unexpected_error_during_encode_is_reraised(processor, monkeypatch):
    def fail_save(self, *_a, **_k):
        raise RuntimeError("disk full")

    data = _encode("PNG")
    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(RuntimeError, match="disk full"):
        processor.process(data)
