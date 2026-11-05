import logging

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# The shared "app" logger (same one utils.logger exposes). Imported via logging
# directly so config stays free of app imports; handlers attach once app_logging.setup runs.
logger = logging.getLogger("app")
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
 
    opensearch_host: str
    opensearch_port: int
    opensearch_user: str
    opensearch_password: str
    opensearch_timeout: int
 
    # OCI GenAI (Cohere Embed) — query/document embeddings (all values from .env)
    oci_config_file: str
    oci_profile: str
    oci_compartment_id: str
    oci_region: str
    oci_genai_endpoint: str
 
    embedding_model_name: str
    embedding_dimension: int
    embedding_normalize: bool
 
 
    opensearch_index_name: str = "asmo-search-api-index"
    opensearch_vector_field: str = "content_vector"
    default_top_k: int = 10
    max_top_k: int = 100
 
    hybrid_semantic_weight: float = 0.5
    hybrid_keyword_weight: float = 0.5
    hybrid_retrieval_k: int = 100
    # Text search: return only products that contain the query words (BM25 match); the vector
    # search just re-ranks them instead of padding the results with "similar" but unrelated
    # products (e.g. fans for "pump"). No product contains the words -> 0 results.
    # false = old behaviour: always mix in vector-only matches.
    text_require_keyword_match: bool = True
    # How many of the query words a product must contain (OpenSearch minimum_should_match).
    # "2<75%": 1-2 words -> all of them; 3+ words -> 75% (e.g. 2 of 3). "" = any one word is enough.
    text_min_words_match: str = "2<75%"
    # HNSW candidate list size per kNN query. The Lucene engine ignores the index
    # setting knn.algo_param.ef_search (it searches only ~k candidates), which missed
    # exact image matches at k=100; 500 found every known-answer test image.
    knn_ef_search: int = 500
    # Image matches scoring below this are dropped. Scores are (1 + cosine) / 2,
    # so an exact copy is 1.0; 0 disables. Calibrate with eval_image_search.py.
    image_min_score: float = 0.6
    # Images whose pixels vary less than this (std dev per colour channel, 0-255) are treated as
    # blank / single colour and not searched. 0 = off.
    image_min_detail: float = 6.0
    # Weight of the image kNN in text + image RRF fusion; the rest is split
    # equally between BM25 and the text kNN. 0.4 = text 60% (BM25 30% + text kNN 30%), image 40%.
    text_image_image_weight: float = 0.4
    # Text shorter than this many characters is not searched (search and suggestions).
    min_query_chars: int = 3
    # Uploaded images larger than this many pixels are scaled down (aspect ratio
    # kept) before embedding; see services/image_processor.py.
    image_max_pixels: int = 2_400_000
    # JPEG quality used when re-encoding a preprocessed (non-PNG) upload.
    image_jpeg_quality: int = 90

 
    # Redis — search-context store for /v1/search. Each search is one key, its
    # request_id, holding {"text_emb", "image_emb", "query"}; it expires
    # redis_ttl_seconds after it was last used.
    redis_enabled: bool = True
    redis_host: str = ""
    redis_port: int = 443
    redis_username: str = "default"
    redis_password: str = ""
    redis_db: int = 0
    redis_ssl: bool = True
    redis_ssl_cert_reqs: str = "none"
    redis_timeout: int = 10
    redis_ttl_seconds: int = 3600

    # Logging — every log line always prints to the console; LOG_TO_FILE=true
    # also writes JSON lines to logs/asmo-<YYYY-MM-DD-HH-MM-SS>.log (a new file for every app start).
    log_level: str = "INFO"
    log_to_file: bool = False
    # Azure Application Insights: when set, logs are also sent there.
    app_insights_connection_string: str = ""
    # Environment name written on every log line (e.g. dev / qa / prod).
    env_name: str | None = None

    # How often (seconds) to send a tiny embed to OCI GenAI so the model stays warm. Each tick is a billable call.
    oci_embed_heartbeat_interval: int = 60
    # Auto-reload on code changes when started with `python app.py` (local dev only; never in a OneDrive folder).
    reload: bool = False

    # Keycloak — checks the "Authorization: Bearer <token>" on every endpoint (utils/keycloak_auth.py).
    keycloak_auth_enabled: bool = True
    keycloak_server_url: str = ""
    keycloak_realm: str = ""
    keycloak_client_id: str = ""
    keycloak_client_secret: str = ""
    keycloak_timeout: int = 5

    api_title: str = "ASMO AI Search API"
    api_version: str = "2.1.0"
    api_description: str = "Frontend contract v2.1 — text/image/filter search with hybrid BM25 + kNN (RRF) and suggestions"
 
 
# Settings are read once, when this module is first imported.
try:
    settings = Settings()
except ValidationError as exc:
    # Log field names + error types only: pydantic's own message includes the input values, which may be secrets.
    problems = ", ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in exc.errors())
    logger.error(
        f"[config] invalid/missing settings ({exc.error_count()}): {problems}",
        extra={"function_name": "config", "status": "error", "error_details": problems},
    )
    raise