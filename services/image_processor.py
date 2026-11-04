"""Image processing for the search upload: validate the multipart image part, then prepare it for embedding.

One entry point, :func:`process_image`, runs the whole flow in order:

  1. read the part     — a file part (UploadFile); a plain text field (e.g. Base64) has no Content-Type
  2. Content-Type      — must be image/jpeg, image/png or image/webp           -> else 400
  3. size              — at most MAX_IMAGE_BYTES (10 MiB)                       -> else 413
  4. magic bytes       — the bytes must really be a JPEG, PNG or WebP (GIF too) -> else 400
  5. decode            — Pillow must be able to open it                         -> else 400
  6. EXIF orientation  — apply the camera rotation, if any
  7. downscale         — above IMAGE_MAX_PIXELS, shrink (aspect ratio kept)
  8. re-encode         — PNG stays PNG, everything else becomes RGB JPEG

It is synchronous on purpose (Pillow work is CPU-heavy): the caller runs it with
``await asyncio.to_thread(process_image, part)`` so the event loop stays free.
"""
from __future__ import annotations

import math
import time
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from config import settings
from models.response import ApiError
from utils.logger import logger

# Image rules (agreed with the frontend).
IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB (10485760 bytes)
NOT_AN_IMAGE_PART = (
    "The image must be sent as binary in a multipart file part with Content-Type "
    "image/jpeg, image/png or image/webp."
)
UNSUPPORTED_IMAGE = "Unsupported image type. Only JPEG, PNG and WebP images are accepted."


def process_image(part: UploadFile | str | None) -> tuple[bytes, str] | None:
    """Validate and prepare the uploaded ``image`` part.

    Returns ``(processed_bytes, mime_type)`` ready for the embedder, or None when no
    (or an empty) image was sent. Raises :class:`ApiError` (400 UNSUPPORTED_IMAGE_TYPE
    or 413 IMAGE_TOO_LARGE) for an image that breaks the rules.
    """
    start = time.perf_counter()

    # 1. Read the part. A part without a filename arrives as a text field -> no declared type.
    if isinstance(part, UploadFile):
        part.file.seek(0)
        data = part.file.read()
        declared = (part.content_type or "").split(";")[0].strip().lower()
    elif isinstance(part, str) and part:
        data, declared = part.encode("utf-8"), None
    else:
        data, declared = b"", None
    if not data:
        logger.info("Image — no image sent")
        return None
    logger.info(f"Image — received {len(data)} bytes, declared Content-Type={declared!r}, "
                f"filename={getattr(part, 'filename', None)!r}")

    # 2. Content-Type. Another image type (e.g. image/gif) is "unsupported"; anything else is "not an image part".
    if declared not in IMAGE_CONTENT_TYPES:
        logger.warning(f"Image — rejected part with Content-Type {declared!r}")
        is_other_image = declared is not None and declared.startswith("image/")
        raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", UNSUPPORTED_IMAGE if is_other_image else NOT_AN_IMAGE_PART)

    # 3. Size.
    if len(data) > MAX_IMAGE_BYTES:
        logger.warning(f"Image — too large: {len(data)} bytes")
        raise ApiError(413, "IMAGE_TOO_LARGE", f"Image exceeds the maximum size of {MAX_IMAGE_BYTES} bytes.")

    # 4. Magic bytes: the bytes decide the format, whatever the declared type.
    is_png = data.startswith(b"\x89PNG")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (is_png or is_jpeg or is_webp):
        logger.warning("Image — bytes are not a JPEG/PNG/WebP image")
        raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", UNSUPPORTED_IMAGE)

    # 5. Decode, and 6. apply the EXIF rotation.
    try:
        image = Image.open(BytesIO(data))
        original_format = image.format
        image = ImageOps.exif_transpose(image)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
        logger.warning(f"Image — could not decode upload ({type(exc).__name__}): {exc}")
        raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", UNSUPPORTED_IMAGE) from exc

    # 7. Downscale large images.
    width, height = image.size
    logger.info(f"Image — decoded {original_format} {width}x{height}")
    if width * height > settings.image_max_pixels:
        scale = math.sqrt(settings.image_max_pixels / (width * height))
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        logger.info(f"Image — downscaled {width}x{height} -> {image.size[0]}x{image.size[1]}")

    # 8. Re-encode.
    output = BytesIO()
    if original_format == "PNG":
        image.save(output, format="PNG")
        mime = "image/png"
    else:
        image.convert("RGB").save(output, format="JPEG", quality=settings.image_jpeg_quality)
        mime = "image/jpeg"
    processed = output.getvalue()

    logger.info(f"Image — processed {len(data)} -> {len(processed)} bytes ({mime}) "
                 f"in {(time.perf_counter() - start) * 1000:.1f}ms")
    return processed, mime
