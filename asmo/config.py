import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """All CommerceTools connection settings in one place.

    Reads credentials from environment variables; call load_dotenv() before
    instantiating (done in main.py) so .env values are present.
    """

    client_id: str = field(default_factory=lambda: os.environ["CT_CLIENT_ID"])
    client_secret: str = field(default_factory=lambda: os.environ["CT_CLIENT_SECRET"])
    project_key: str = field(default_factory=lambda: os.environ.get("CT_PROJECT_KEY"))
    region: str = field(default_factory=lambda: os.environ.get("CT_REGION"))
    channel_id: str = field(default_factory=lambda: os.environ["CT_CHANNEL_ID"])
    upload_enabled: bool = field(default_factory=lambda: os.environ.get("UPLOAD_ENABLED", "false").lower() == "true")
    page_size: int = 500

    @property
    def auth_url(self) -> str:
        return f"https://auth.{self.region}.commercetools.com/oauth/token"

    @property
    def api_base(self) -> str:
        return f"https://api.{self.region}.commercetools.com/{self.project_key}"

    @property
    def scope(self) -> str:
        scopes = [
            "manage_project",
            "view_audit_log",
        ]
        return " ".join(f"{s}:{self.project_key}" for s in scopes)
