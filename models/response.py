from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    id: str = Field(description="OpenSearch document _id")
    index: str = Field(description="Index the document was retrieved from")
    score: float = Field(description="Raw relevance score assigned by OpenSearch or the reranker")
    source: dict[str, Any] = Field(description="Full document source as stored in the index")
    highlights: dict[str, list[str]] | None = Field(
        default=None,
        description="Per-field highlighted snippets returned by OpenSearch highlight query",
    )


class FacetTermValue(BaseModel):
    value: str = Field(description="Facet bucket value")
    count: int = Field(description="Number of documents matching this value")


class TermsFacet(BaseModel):
    key: str = Field(description="Filter key that corresponds to this facet, e.g. category_ids")
    type: Literal["terms"] = Field(default="terms", description="Discriminator — always 'terms' for this variant")
    values: list[FacetTermValue] = Field(description="Ordered list of term buckets with their document counts")


class RangeFacet(BaseModel):
    key: str = Field(description="Filter key that corresponds to this facet, e.g. price")
    type: Literal["range"] = Field(default="range", description="Discriminator — always 'range' for this variant")
    min: float = Field(description="Minimum value of this field across the result set")
    max: float = Field(description="Maximum value of this field across the result set")
    currency: str | None = Field(default=None, description="ISO 4217 currency code when the field represents a monetary amount")


Facet = Annotated[Union[TermsFacet, RangeFacet], Field(discriminator="type")]


class ProductResult(BaseModel):
    id: str = Field(description="Product document ID in OpenSearch")
    score: float = Field(description="Final relevance score after RRF fusion and cross-encoder reranking")
    source: dict[str, Any] = Field(description="Raw product document fields as stored in the index")


class SearchApiResponse(BaseModel):
    request_id: str | None = Field(default=None, description="UUID generated per request, useful for tracing and support")
    image_ref: str | None = Field(default=None, description="Echoed back from the request when an image reference was provided")
    search_type: str = Field(description="Indicates the search mode used: 'hybrid' for text queries, 'filter' for browse-only")
    query: str | None = Field(default=None, description="The original query string submitted by the client")
    total: int = Field(description="Total number of matching documents in the index before pagination")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of results returned on this page")
    total_pages: int = Field(description="Total number of pages available given the page_size")
    took_ms: int = Field(description="Time in milliseconds spent executing the query in OpenSearch")
    results: list[ProductResult] = Field(description="Ranked product results for the requested page")
    facets: list[Facet] = Field(default=[], description="Aggregated facets computed across the full result set, ordered by configuration")


class ProductSuggestion(BaseModel):
    text: str = Field(description="Suggested product name to display in the autocomplete dropdown")
    sku: str = Field(description="SKU of the suggested product")


class CategorySuggestion(BaseModel):
    text: str = Field(description="Category or sub-category name matching the typed prefix")


class SuggestionsResponse(BaseModel):
    query: str = Field(description="The prefix string that was used to generate these suggestions")
    product_suggestions: list[ProductSuggestion] = Field(description="Product name suggestions matching the prefix")
    category_suggestions: list[CategorySuggestion] = Field(description="Category and sub-category suggestions matching the prefix")


class IndexInfo(BaseModel):
    name: str = Field(description="Index name in OpenSearch")
    doc_count: int = Field(description="Number of documents currently in the index")
    size_bytes: int = Field(description="Total on-disk size of the index in bytes")
    status: str = Field(description="Index health status as reported by OpenSearch, e.g. green, yellow, red")


class IndicesResponse(BaseModel):
    indices: list[IndexInfo] = Field(description="List of non-system indices available in the cluster")
    total: int = Field(description="Total number of indices returned")


class HealthResponse(BaseModel):
    status: str = Field(description="Overall API health: 'ok' when all dependencies are reachable, 'degraded' otherwise")
    opensearch_connected: bool = Field(description="Whether the API successfully pinged the OpenSearch cluster")
    opensearch_cluster: str | None = Field(default=None, description="Cluster name returned by OpenSearch, null if unreachable")
    embedding_model_loaded: bool = Field(description="Whether the sentence-transformer embedding model has been loaded into memory")
    embedding_model_name: str = Field(description="HuggingFace model identifier used for query embedding")


class ErrorDetail(BaseModel):
    code: str = Field(description="Machine-readable error code, e.g. INVALID_FILTER_KEY, EMPTY_REQUEST")
    message: str = Field(description="Human-readable description of what went wrong")
    detail: Any | None = Field(default=None, description="Optional additional context, such as the offending value or field name")


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(description="Structured error payload returned on 4xx and 5xx responses")
