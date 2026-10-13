import asyncio
import aioboto3
import json
import os
from pathlib import Path
from botocore.client import Config
from typing import Optional
from enum import Enum
from .logging_service import logger


class Folder(str, Enum):
    PRODUCTS = "products-data"
    CATEGORY = "category-data"


class ObjectStorageConfig:
    """Configuration for OpenShift ODF Object Storage."""

    def __init__(self):
        self.endpoint_url = f"https://{os.environ['S3_HOST']}:{os.environ.get('S3_PORT', '443')}"
        self.bucket_name = os.environ["S3_BUCKET_NAME"]
        self.access_key = os.environ["S3_ACCESS_KEY"]
        self.secret_key = os.environ["S3_SECRET_KEY"]
        self.verify_ssl = os.environ.get("S3_VERIFY_SSL", "false").lower() == "true"


class LocalFolderReader:
    """Reads JSON files from a local directory."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self._validate()

    def _validate(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Local folder not found: {self.folder_path}")
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.folder_path}")

    def get_json_files(self) -> list[Path]:
        files = sorted(self.folder_path.glob("*.json"))
        if not files:
            logger.warning(f"No JSON files found in '{self.folder_path}'")
        else:
            logger.info(f"Found {len(files)} JSON file(s) in '{self.folder_path}'")
        return files

    def read_file(self, file_path: Path) -> bytes:
        return file_path.read_bytes()

    def read_and_validate_file(self, file_path: Path) -> bytes:
        raw = file_path.read_bytes()
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in '{file_path.name}': {e}")
        return raw

    def load_all(self, validate: bool = True) -> list[tuple[str, bytes]]:
        files = self.get_json_files()
        result = []
        for file_path in files:
            try:
                raw = self.read_and_validate_file(file_path) if validate else self.read_file(file_path)
                result.append((file_path.name, raw))
                logger.info(f"Read '{file_path.name}' ({len(raw)} bytes)")
            except (ValueError, OSError) as e:
                logger.error(f"Skipping '{file_path.name}': {e}")
        return result


class ObjectStorageClient:
    """Async S3-compatible client for OpenShift ODF Object Storage."""

    def __init__(self, config: ObjectStorageConfig):
        self.config = config
        self._session = aioboto3.Session()

    def _client_kwargs(self) -> dict:
        return {
            "service_name": "s3",
            "endpoint_url": self.config.endpoint_url,
            "aws_access_key_id": self.config.access_key,
            "aws_secret_access_key": self.config.secret_key,
            "config": Config(signature_version="s3v4"),
            "verify": self.config.verify_ssl,
        }

    @staticmethod
    def _build_key(folder: Folder, filename: str) -> str:
        return f"{folder.value}/{filename}"

    # Bucket 

    async def bucket_exists(self) -> bool:
        async with self._session.client(**self._client_kwargs()) as s3:
            try:
                await s3.head_bucket(Bucket=self.config.bucket_name)
                return True
            except Exception as exc:
                logger.warning(
                    f"bucket_exists check failed for '{self.config.bucket_name}': {exc}",
                    extra={"function_name": "bucket_exists", "status": "error", "error_details": str(exc)},
                )
                return False

    async def list_folder(self, folder: Folder) -> list[dict]:
        async with self._session.client(**self._client_kwargs()) as s3:
            response = await s3.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=f"{folder.value}/",
            )
            objects = response.get("Contents", [])
            logger.info(f"[{folder.value}] Found {len(objects)} object(s)")
            return objects

    async def list_all_folders(self) -> dict[str, list[dict]]:
        results = await asyncio.gather(*[self.list_folder(folder) for folder in Folder])
        return {folder.value: result for folder, result in zip(Folder, results)}

    # Upload 

    async def upload_file(self, folder: Folder, local_path: str, filename: str) -> None:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            await s3.upload_file(local_path, self.config.bucket_name, s3_key)
            logger.info(f"[{folder.value}] Uploaded '{local_path}' -> '{s3_key}'")

    async def upload_bytes(
        self,
        folder: Folder,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(f"[{folder.value}] Uploaded {len(data)} bytes -> '{s3_key}'")

    async def upload_many(self, folder: Folder, files: list[tuple[str, bytes, str]]) -> None:
        await asyncio.gather(
            *[self.upload_bytes(folder, filename, data, content_type)
              for filename, data, content_type in files]
        )

    async def upload_from_local_folder(
        self,
        local_folder: str,
        target_folder: Folder,
        validate_json: bool = True,
        concurrency: int = 10,
    ) -> dict[str, int]:
        reader = LocalFolderReader(local_folder)
        files = reader.load_all(validate=validate_json)

        if not files:
            logger.warning("No files to upload. Aborting.")
            return {"uploaded": 0, "failed": 0}

        semaphore = asyncio.Semaphore(concurrency)
        uploaded, failed = 0, 0

        async def _upload_one(filename: str, data: bytes):
            nonlocal uploaded, failed
            async with semaphore:
                try:
                    await self.upload_bytes(
                        folder=target_folder,
                        filename=filename,
                        data=data,
                        content_type="application/json",
                    )
                    uploaded += 1
                except Exception as e:
                    logger.error(f"Failed to upload '{filename}': {e}")
                    failed += 1

        await asyncio.gather(*[_upload_one(name, data) for name, data in files])

        logger.info(
            f"[{target_folder.value}] Upload complete — {uploaded} succeeded, {failed} failed"
        )
        return {"uploaded": uploaded, "failed": failed}


    async def download_file(self, folder: Folder, filename: str, local_path: str) -> None:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            await s3.download_file(self.config.bucket_name, s3_key, local_path)
            logger.info(f"[{folder.value}] Downloaded '{s3_key}' -> '{local_path}'")

    async def read_object(self, folder: Folder, filename: str) -> bytes:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            response = await s3.get_object(Bucket=self.config.bucket_name, Key=s3_key)
            content = await response["Body"].read()
            logger.info(f"[{folder.value}] Read {len(content)} bytes from '{s3_key}'")
            return content

    async def read_many(self, folder: Folder, filenames: list[str]) -> dict[str, bytes]:
        contents = await asyncio.gather(
            *[self.read_object(folder, filename) for filename in filenames]
        )
        return dict(zip(filenames, contents))

    # Delete 

    async def delete_object(self, folder: Folder, filename: str) -> None:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            await s3.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
            logger.info(f"[{folder.value}] Deleted '{s3_key}'")

    async def delete_many(self, folder: Folder, filenames: list[str]) -> None:
        if not filenames:
            return
        objects = [{"Key": self._build_key(folder, f)} for f in filenames]
        async with self._session.client(**self._client_kwargs()) as s3:
            response = await s3.delete_objects(
                Bucket=self.config.bucket_name,
                Delete={"Objects": objects, "Quiet": True},
            )
            errors = response.get("Errors", [])
            if errors:
                logger.error(f"[{folder.value}] Batch delete errors: {errors}")
            else:
                logger.info(f"[{folder.value}] Batch deleted {len(filenames)} object(s)")

    async def delete_folder(self, folder: Folder) -> None:
        objects = await self.list_folder(folder)
        filenames = [obj["Key"].removeprefix(f"{folder.value}/") for obj in objects]
        await self.delete_many(folder, filenames)
        logger.info(f"[{folder.value}] Folder cleared")


    async def object_exists(self, folder: Folder, filename: str) -> bool:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            try:
                await s3.head_object(Bucket=self.config.bucket_name, Key=s3_key)
                return True
            except Exception as exc:
                logger.info(
                    f"object_exists check failed for '{s3_key}': {exc}",
                    extra={"function_name": "object_exists", "status": "error", "error_details": str(exc), "input_data": {"key": s3_key}},
                )
                return False

    async def object_metadata(self, folder: Folder, filename: str) -> Optional[dict]:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            try:
                return await s3.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            except Exception as exc:
                logger.info(
                    f"object_metadata lookup failed for '{s3_key}': {exc}",
                    extra={"function_name": "object_metadata", "status": "error", "error_details": str(exc), "input_data": {"key": s3_key}},
                )
                return None

    async def generate_presigned_url(
        self, folder: Folder, filename: str, expires_in: int = 3600
    ) -> str:
        s3_key = self._build_key(folder, filename)
        async with self._session.client(**self._client_kwargs()) as s3:
            return await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": s3_key},
                ExpiresIn=expires_in,
            )
