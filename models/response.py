from __future__ import annotations

from pydantic import BaseModel, Field


# Search API — response contract (ASMO AI Search API Specification v2.1, §2.4)


class SellerDetail(BaseModel):
    """One seller offer for the SKU (source: seller_details[])."""

    seller_id: str | None = Field(default=None, description="Seller identifier (source: seller_details[].seller_id)")
    name: str | None = Field(default=None, description="Seller name (source: seller_details[].seller_name)")
    lead_time: int | None = Field(default=None, description="Lead time in days (source: seller_details[].lead_time)")
    unit_type: str | None = Field(default=None, description="Unit of measure, e.g. Ea (source: seller_details[].unit_type)")
    price: float | None = Field(default=None, description="Unit price (source: seller_details[].price)")
    moq: int | str = Field(default="", description="Minimum order quantity (source: seller_details[].moq); \"\" when the seller has none")
    currency: str | None = Field(default=None, description="ISO 4217 currency code (source: seller_details[].currency)")
    temporary_unavailable: bool | None = Field(default=None, description="Seller offer temporarily unavailable (source: seller_details[].temporary_unavailable)")


class ProductItem(BaseModel):
    """A single product in the search response, shaped per the agreed frontend contract.

    The primary (cheapest) seller is ``seller_details[0]``; its price is what
    ``sort_by=price`` orders by.
    """

    sku_id: str | None = Field(default=None, description="SKU identifier (source: sku_id)")
    sku_name: str | None = Field(default=None, description="Product name (source: sku_name, else product_name)")
    primary_image_url: str | None = Field(default=None, description="Primary product image URL (source: primary_image_url)")
    asmo_material_number: str | None = Field(default=None, description="ASMO material number (source: asmo_material_number)")
    aramco_material_number: str | None = Field(default=None, description="Aramco material number (source: aramco_material_number)")
    seller_details: list[SellerDetail] = Field(default_factory=list, description="Every seller offer, cheapest first (source: seller_details)")


class FacetTermValue(BaseModel):
    id: str = Field(description="Value to send back as the filter for this facet (the category / sub-category / product-type / seller id, or the material type)")
    value: str = Field(description="Facet bucket label to display")
    count: int = Field(description="Number of matching products with this value")


class PriceFacet(BaseModel):
    min: float | None = Field(default=None, description="Minimum price across the matching result population")
    max: float | None = Field(default=None, description="Maximum price across the matching result population")


class SearchFacets(BaseModel):
    """Facets keyed by field, computed over the full matching population (§2.4 facet rule)."""

    category_name: list[FacetTermValue] = Field(default_factory=list, description="Categories (id = category id; filter key category_name)")
    sub_category_name: list[FacetTermValue] = Field(default_factory=list, description="Sub-categories (id = sub-category id; filter key sub_category_name)")
    product_type: list[FacetTermValue] = Field(default_factory=list, description="Product types (id = product type id; filter key product_type)")
    sellername: list[FacetTermValue] = Field(default_factory=list, description="Sellers (id = seller id; filter key sellername)")
    material_type: list[FacetTermValue] = Field(default_factory=list, description="Material types (id = the value; filter key material_type)")
    price: PriceFacet | None = Field(default=None, description="Price range (min/max) across the result set")


class SearchApiResponse(BaseModel):
    request_id: str | None = Field(default=None, description="Search-context handle: send it back as request_id (with filters / sort / page) to refine this search. Refining returns the same request_id; a new query or image gets a new one")
    total_results: int = Field(description="Total number of matching products before pagination")
    page: int = Field(description="Current 1-based page number")
    page_size: int = Field(description="Number of products returned per page")
    products: list[ProductItem] = Field(description="Products for the requested page")
    facets: SearchFacets = Field(default_factory=SearchFacets, description="Facets computed across the full result set")



# Suggestions API — response contract (§3.2)



class QuerySuggestion(BaseModel):
    text: str = Field(description="Suggested search query string matching the prefix")


class ProductSuggestion(BaseModel):
    text: str = Field(description="Suggested product name to display in the autocomplete dropdown")
    sku_id: str | None = Field(default=None, description="SKU of the suggested product, to link to its detail page (source: sku_id)")


class SuggestionsResponse(BaseModel):
    query: str = Field(description="The prefix string used to generate these suggestions")
    query_suggestions: list[QuerySuggestion] = Field(description="Suggested search query strings matching the prefix")
    product_suggestions: list[ProductSuggestion] = Field(description="Product suggestions matching the prefix")



# Operational / diagnostic models (not part of the frontend contract)



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
    embedding_model_loaded: bool = Field(description="Whether the embedding model client has been initialized")
    embedding_model_name: str = Field(description="OCI GenAI model identifier used for query embedding")
    redis_connected: bool | None = Field(default=None, description="Whether Redis (embedding cache) answered PING; null when Redis is disabled")


class LatencyResponse(BaseModel):
    target: str = Field(description="Which dependency was measured: 'opensearch', 'redis' or 'oci'")
    endpoint: str = Field(description="Host/endpoint that was contacted")
    samples: int = Field(description="Number of round-trips attempted")
    samples_ms: list[float] = Field(description="Per-call round-trip time in milliseconds, in order (first is usually cold)")
    min_ms: float = Field(description="Fastest round-trip (best-case / warm connection)")
    avg_ms: float = Field(description="Average round-trip across samples")
    max_ms: float = Field(description="Slowest round-trip (usually the cold/first call)")
    ok: bool = Field(description="True if all round-trips succeeded")
    error: str | None = Field(default=None, description="Error message if a round-trip failed")



# Error contract (§5.1) — flat {code, message}



class ErrorResponse(BaseModel):
    code: str = Field(description="Stable, machine-readable backend error identifier, e.g. INVALID_FILTER_KEY")
    message: str = Field(description="Human-readable description suitable for frontend handling or display")
