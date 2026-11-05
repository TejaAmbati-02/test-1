"""Image processing for the search upload: validate the multipart image part and prepare it for embedding.

Use the ready-made object:

    from services.image_processor import image_processor
    image, blank = await asyncio.to_thread(image_processor.process, part)

``process`` is a normal (sync) method on purpose: Pillow work is CPU-heavy, so the caller runs it in
a worker thread to keep the event loop free. It is self-contained — it calls no other function of
this module — so it can be copied into another code base as is.
"""
import math
import time
from io import BytesIO

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
from starlette.datastructures import UploadFile

from config import settings
from models.response import ApiError
from utils.logger import logger


class ImageProcessor:
    def __init__(self, max_bytes: int, max_pixels: int, jpeg_quality: int, min_detail: float) -> None:
        self.max_bytes = max_bytes        # bigger uploads -> 413
        self.max_pixels = max_pixels      # bigger images are scaled down
        self.jpeg_quality = jpeg_quality  # quality when re-encoding as JPEG
        self.min_detail = min_detail      # pixel variation below this = blank image (0 = off)

    def process(self, part: UploadFile | str | None) -> tuple[tuple[bytes, str] | None, bool]:
        """Validate and prepare the uploaded ``image`` part.

        Returns ``(image, blank)``:
          * ``((processed_bytes, mime_type), False)`` — a usable image, ready for the embedder
          * ``(None, False)`` — no (or an empty) image was sent
          * ``(None, True)``  — the image is blank / a single colour: nothing to search for

        Raises ApiError 400 UNSUPPORTED_IMAGE_TYPE or 413 IMAGE_TOO_LARGE for an image that breaks the rules.
        """
        start = time.perf_counter()
        unsupported = "Unsupported image type. Only JPEG, PNG and WebP images are accepted."

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
            return None, False
        logger.info(f"Image — received {len(data)} bytes, declared Content-Type={declared!r}, "
                    f"filename={getattr(part, 'filename', None)!r}")

        # 2. Content-Type must be JPEG / PNG / WebP. Another image type (e.g. image/gif) is "unsupported";
        #    anything else (a text field, application/octet-stream, ...) is "not an image part".
        if declared not in ("image/jpeg", "image/png", "image/webp"):
            logger.warning(f"Image — rejected part with Content-Type {declared!r}")
            if declared is not None and declared.startswith("image/"):
                raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", unsupported)
            raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", "The image must be sent as binary in a multipart file part "
                                                          "with Content-Type image/jpeg, image/png or image/webp.")

        # 3. Size.
        if len(data) > self.max_bytes:
            logger.warning(f"Image — too large: {len(data)} bytes")
            raise ApiError(413, "IMAGE_TOO_LARGE", f"Image exceeds the maximum size of {self.max_bytes} bytes.")

        # 4. Magic bytes: the bytes decide the format, whatever the declared type.
        is_png = data.startswith(b"\x89PNG")
        is_jpeg = data.startswith(b"\xff\xd8\xff")
        is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        if not (is_png or is_jpeg or is_webp):
            logger.warning("Image — bytes are not a JPEG/PNG/WebP image")
            raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", unsupported)

        # 5. Decode, and apply the camera's EXIF rotation.
        try:
            image = Image.open(BytesIO(data))
            original_format = image.format
            image = ImageOps.exif_transpose(image)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as exc:
            logger.warning(f"Image — could not decode upload ({type(exc).__name__}): {exc}")
            raise ApiError(400, "UNSUPPORTED_IMAGE_TYPE", unsupported) from exc
        width, height = image.size
        logger.info(f"Image — decoded {original_format} {width}x{height}")

        # 6. Blank / single-colour image? Measure how much the pixels vary (standard deviation per
        #    colour channel, 0-255): a plain colour is ~0, a real product photo is far higher.
        detail = max(ImageStat.Stat(image.convert("RGB")).stddev)
        if detail < self.min_detail:
            logger.info(f"Image — blank / single colour (detail={detail:.1f} < {self.min_detail}) -> not searched")
            return None, True
        logger.info(f"Image — detail={detail:.1f} (minimum {self.min_detail})")

        # 7. Scale large images down (aspect ratio kept).
        if width * height > self.max_pixels:
            scale = math.sqrt(self.max_pixels / (width * height))
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
            logger.info(f"Image — downscaled {width}x{height} -> {image.size[0]}x{image.size[1]}")

        # 8. Re-encode: PNG stays PNG, everything else becomes an RGB JPEG.
        output = BytesIO()
        if original_format == "PNG":
            image.save(output, format="PNG")
            mime = "image/png"
        else:
            image.convert("RGB").save(output, format="JPEG", quality=self.jpeg_quality)
            mime = "image/jpeg"
        processed = output.getvalue()

        logger.info(f"Image — processed {len(data)} -> {len(processed)} bytes ({mime}) "
                    f"in {(time.perf_counter() - start) * 1000:.1f}ms")
        return (processed, mime), False


# The one object the app uses — import it where needed:  from services.image_processor import image_processor
image_processor = ImageProcessor(
    max_bytes=10 * 1024 * 1024,  # 10 MiB (agreed with the frontend)
    max_pixels=settings.image_max_pixels,
    jpeg_quality=settings.image_jpeg_quality,
    min_detail=settings.image_min_detail,
)
