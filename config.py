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
    # HNSW candidate list size per kNN query. The Lucene engine ignores the index
    # setting knn.algo_param.ef_search (it searches only ~k candidates), which missed
    # exact image matches at k=100; 500 found every known-answer test image.
    knn_ef_search: int = 500
    # Image matches scoring below this are dropped. Scores are (1 + cosine) / 2,
    # so an exact copy is 1.0; 0 disables. Calibrate with eval_image_search.py.
    image_min_score: float = 0.0
    # Weight of the image kNN in text + image RRF fusion; the rest is split
    # equally between BM25 and the text kNN.
    text_image_image_weight: float = 0.6
 
 
    # Redis — embedding cache for /v1/search. Each query / image embedding is an
    # entry in one Redis hash (the collection) keyed by its embedding id, and
    # expires redis_ttl_seconds after it is written.
    redis_enabled: bool = True
    redis_host: str = ""
    redis_port: int = 443
    redis_username: str = "default"
    redis_password: str = ""
    redis_db: int = 0
    redis_ssl: bool = True
    redis_ssl_cert_reqs: str = "none"
    redis_timeout: int = 10
    redis_collection: str = "asmo:embeddings"
    redis_ttl_seconds: int = 3600

    # Logging — every log line always prints to the console; LOG_TO_FILE=true
    # also writes JSON lines to logs/asmo-<timestamp>.log.
    log_level: str = "INFO"
    log_to_file: bool = False

    api_title: str = "ASMO AI Search API"
    api_version: str = "2.1.0"
    api_description: str = "Frontend contract v2.1 — text/image/filter search with hybrid BM25 + kNN (RRF) and suggestions"
 
 
settings = Settings()