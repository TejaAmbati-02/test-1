import hashlib
import ssl

import truststore
from fastapi import Request
from keycloak import KeycloakOpenID
from keycloak.exceptions import KeycloakConnectionError, KeycloakError

from config import settings
from models.response import ApiError
from utils.logger import logger


class KeycloakAuth:
    def __init__(self) -> None:
        # Settings come from .env through config.py
        self.enabled = settings.keycloak_auth_enabled
        self.client = KeycloakOpenID(
            server_url=settings.keycloak_server_url,
            realm_name=settings.keycloak_realm,
            client_id=settings.keycloak_client_id,
            client_secret_key=settings.keycloak_client_secret,
            timeout=settings.keycloak_timeout,
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),  # trust the company's certificates
        )

    async def require_auth(self, request: Request) -> None:
        """Let the request through only if it carries a valid Keycloak access token.

        Use on an endpoint: @router.get("/items", dependencies=[Depends(keycloak_auth.require_auth)])
        On success the user's details are saved on request.state.user.
        """
        # 1. Auth switched off (local dev / tests) -> let everything through.
        if not self.enabled:
            return

        # 2. Read the token from the "Authorization: Bearer <token>" header.
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            logger.warning(f"[Keycloak] {request.method} {request.url.path}: missing or malformed Authorization header")
            raise ApiError(401, "UNAUTHORIZED", "Send 'Authorization: Bearer <access_token>'",
                           {"WWW-Authenticate": 'Bearer realm="api"'})

        # A short fingerprint of the token for the logs (sha256, first 12 chars). The same token
        # always gives the same hash, so its requests can be traced — the token itself is never logged.
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:12]

        # 3. Ask Keycloak whether the token is still valid.
        try:
            user = await self.client.a_introspect(token)
        except KeycloakConnectionError:
            logger.error(f"[Keycloak] token {token_hash}: check failed, Keycloak is not reachable")
            raise ApiError(503, "AUTH_UNAVAILABLE", "Keycloak is not reachable")
        except KeycloakError as exc:
            # Keycloak refused *this app* (wrong KEYCLOAK_CLIENT_ID / SECRET), not the user's token.
            logger.error(f"[Keycloak] token {token_hash}: check failed, Keycloak returned HTTP {exc.response_code}")
            raise ApiError(502, "AUTH_SERVER_ERROR", "Keycloak rejected this app's client id or secret")

        # 4. Expired, logged out or fake token -> 401.
        if not isinstance(user, dict) or user.get("active") is not True:
            logger.warning(f"[Keycloak] token {token_hash}: invalid or expired ({request.method} {request.url.path})")
            raise ApiError(401, "UNAUTHORIZED", "Invalid or expired access token",
                           {"WWW-Authenticate": 'Bearer realm="api"'})

        # 5. Valid -> remember who the user is and let the endpoint run.
        logger.info(f"[Keycloak] token {token_hash}: valid for '{user.get('preferred_username', user.get('sub'))}'")
        request.state.user = user

    async def close(self) -> None:
        """Close the connection to Keycloak (called on app shutdown)."""
        await self.client.connection.aclose()


# The one object the app uses — import it where needed:  from utils.keycloak_auth import keycloak_auth
keycloak_auth = KeycloakAuth()
