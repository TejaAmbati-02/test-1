"""Preprocessing for uploaded search images, applied before they are embedded.

Steps: fix EXIF orientation, scale down images above ``image_max_pixels``
(aspect ratio kept), then re-encode — PNG stays PNG, everything else becomes
RGB JPEG. Works on the raw upload bytes; the embedder does the Base64 step.
"""
from __future__ import annotations

import math
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from config import Settings, settings as _settings
from utils.logger import logger


class InvalidImageError(ValueError):
    """The bytes could not be decoded as an image (corrupt, truncated, or too large to decode safely)."""


class ImageProcessor:
    def __init__(self, max_pixels: int, jpeg_quality: int) -> None:
        self.max_pixels = max_pixels
        self.jpeg_quality = jpeg_quality

    @classmethod
    def from_settings(cls, cfg: Settings) -> "ImageProcessor":
        return cls(max_pixels=cfg.image_max_pixels, jpeg_quality=cfg.image_jpeg_quality)

    def process(self, data: bytes) -> tuple[bytes, str]:
        """Preprocess image bytes and return ``(processed_bytes, mime_type)``.

        Raises:
            InvalidImageError: the bytes are not a decodable image.
        """
        # 1. Read the image from memory.
        try:
            image = Image.open(BytesIO(data))
            original_format = image.format
            # 2. Apply EXIF orientation (rotate/transpose) if present.
            image = ImageOps.exif_transpose(image)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
            raise InvalidImageError(str(exc)) from exc

        # 3. Resize only if the image is too large.
        width, height = image.size
        current_pixels = width * height
        if current_pixels > self.max_pixels:
            scale = math.sqrt(self.max_pixels / current_pixels)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 4. Write the processed image back to memory.
        output = BytesIO()
        if original_format == "PNG":
            image.save(output, format="PNG")
            mime_type = "image/png"
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(output, format="JPEG", quality=self.jpeg_quality)
            mime_type = "image/jpeg"

        processed = output.getvalue()
        logger.debug(
            "ImageProcessor.process — done",
            extra={"input_data": {
                "format": original_format,
                "original_size": [width, height],
                "processed_size": list(image.size),
                "bytes_in": len(data),
                "bytes_out": len(processed),
                "mime": mime_type,
            }},
        )
        return processed, mime_type


image_processor = ImageProcessor.from_settings(_settings)
