import aiohttp
from functools import wraps
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


def requires_token(method):
    """Decorator that auto-fetches a token if one hasn't been acquired yet.

    Works on both AuthClient (checks self._token)
    and any client that holds self._auth (an AuthClient instance).
    """
    @wraps(method)
    async def wrapper(self, *args, **kwargs):
        auth: "AuthClient" = getattr(self, "_auth", self)
        if not auth._token:
            await auth.authenticate()
        return await method(self, *args, **kwargs)
    return wrapper


class AuthClient:
    """Handles OAuth2 client-credentials flow with CommerceTools."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._token: str | None = None


    @log_function
    async def authenticate(self) -> None:
        """Fetch and store a fresh access token."""
        logger.info("Authenticating with CommerceTools", extra={"function_name": "authenticate"})
        async with aiohttp.ClientSession() as session:
            self._token = await self._fetch_token(session)
        logger.info("Token acquired successfully", extra={"function_name": "authenticate", "status": "success"})

    @property
    def token(self) -> str:
        if not self._token:
            raise RuntimeError("Call authenticate() before accessing the token.")
        return self._token

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}


    async def _fetch_token(self, session: aiohttp.ClientSession) -> str:
        from datetime import datetime
        request_data = {"grant_type": "client_credentials", "scope": self._config.scope}
        logger.info(
            f"POST {self._config.auth_url} client_id={self._config.client_id} data={request_data}",
            extra={
                "function_name": "_fetch_token",
                "input_data": {"url": self._config.auth_url, "client_id": self._config.client_id, "data": request_data},
            },
        )
        t0 = datetime.now()
        try:
            async with session.post(
                self._config.auth_url,
                auth=aiohttp.BasicAuth(self._config.client_id, self._config.client_secret),
                data=request_data,
                ssl=SSL_CONTEXT,
            ) as resp:
                full_url = str(resp.url)
                status = resp.status
                body = await resp.json(content_type=None)
                elapsed = (datetime.now() - t0).total_seconds()
                safe_body = {k: ("***" if k == "access_token" else v) for k, v in body.items()}
                logger.info(
                    f"HTTP {status} ← {full_url} in {elapsed:.3f}s (token_type={body.get('token_type')}, expires_in={body.get('expires_in')})",
                    extra={
                        "function_name": "_fetch_token",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "token_type": body.get("token_type"), "expires_in": body.get("expires_in")},
                        "response_payload": safe_body,
                    },
                )
                resp.raise_for_status()
                return body["access_token"]
        except aiohttp.ClientResponseError as e:
            error_msg = body.get("error_description") or body.get("message") or str(e)
            logger.error(
                f"Authentication failed ({e.status}): {error_msg}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": error_msg},
            )
            raise RuntimeError(f"Authentication failed ({e.status}): {error_msg}") from e
        except aiohttp.ClientConnectionError as e:
            logger.error(
                f"Could not reach auth server: {self._config.auth_url}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": str(e)},
            )
            raise RuntimeError(f"Could not reach auth server: {self._config.auth_url}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during authentication: {e}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": str(e)},
            )
            raise
        finally:
            logger.info("Token fetch attempt complete")
