from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
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
 
 
    # Logging — every log line always prints to the console; LOG_TO_FILE=true
    # also writes JSON lines to logs/asmo-<timestamp>.log.
    log_level: str = "INFO"
    log_to_file: bool = False

    api_title: str = "ASMO AI Search API"
    api_version: str = "2.1.0"
    api_description: str = "Frontend contract v2.1 — text/image/filter search with hybrid BM25 + kNN (RRF) and suggestions"
 
 
settings = Settings()