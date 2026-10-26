#!/usr/bin/env bash
# Updated curls for ASMO AI Search API v2.1 — POST /v1/search is now multipart/form-data.
# Set base_url (defaults to localhost:8000). On PowerShell use curl.exe instead of curl.
base_url="${base_url:-http://localhost:8000}"


# ---------- Health & Infra ----------
echo "=== TC-000 Health Check ==="
curl -s "${base_url}/health"

echo "=== TC-000 List Indices ==="
curl -s "${base_url}/indices"


# ---------- Text Search ----------
echo "=== TC-001 Search — pipe ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-002 Search — switchboard ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=switchboard" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-003 Search — pressure transmitter ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pressure transmitter" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-004 Search — centrifugal pump ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=centrifugal pump" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-005 Search — corrosion inhibitor ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=corrosion inhibitor" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-006 Search — fire extinguisher ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=fire extinguisher" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-007 Search — ergonomic chair ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=ergonomic chair" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-008 Search — Arabic language ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=مضخة" \
  -F "language=ar" \
  -F "page=1" \
  -F "page_size=10"


# ---------- Sort ----------
echo "=== TC-009 Sort by price asc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=valve" \
  -F "language=en" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-010 Sort by price desc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=valve" \
  -F "language=en" \
  -F "sort_by=price" \
  -F "sort_order=desc" \
  -F "page=1" \
  -F "page_size=10"


# ---------- Filter Browse ----------
echo "=== TC-011 Browse — Electrical category ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"category_ids": "buyer-default-electrical"}' \
  -F "page=1" \
  -F "page_size=20"

echo "=== TC-012 Browse — Machinery Equipment category ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"category_ids": "buyer-default-machinery-equipment"}' \
  -F "page=1" \
  -F "page_size=20"

echo "=== TC-013 Browse — in stock only ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"availability": true}' \
  -F "page=1" \
  -F "page_size=20"

echo "=== TC-014 Browse — price range 50–150 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"price": {"gte": 50, "lte": 150}}' \
  -F "page=1" \
  -F "page_size=20"

echo "=== TC-015 Browse — customer SHELL ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"customer_id": "SHELL"}' \
  -F "page=1" \
  -F "page_size=20"

echo "=== TC-016 Browse — material type 9CAT ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"material_type": "9CAT"}' \
  -F "page=1" \
  -F "page_size=20"


# ---------- Combined Search + Filters ----------
echo "=== TC-018 Search + availability filter ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=motor control center" \
  -F "language=en" \
  -F 'filters={"availability": true}' \
  -F "page=1" \
  -F "page_size=10"

echo "=== TC-019 Search + price range filter ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=flow meter" \
  -F "language=en" \
  -F 'filters={"price": {"gte": 50, "lte": 200}}' \
  -F "page=1" \
  -F "page_size=10"


# ---------- Specific Field Filters ----------
echo "=== TC-021 Filter by SKU — 012000-005-pipes ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"sku": "012000-005-pipes"}' \
  -F "page=1" \
  -F "page_size=5"

echo "=== TC-023 Filter by hazard health H1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"hazard_health": "H1"}' \
  -F "page=1" \
  -F "page_size=10"


# ---------- Pagination ----------
echo "=== TC-024 Pagination — page 2 differs from page 1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=valve" \
  -F "language=en" \
  -F "page=2" \
  -F "page_size=5"

echo "=== TC-025 Pagination — page 1 (verify differs from page 2) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=valve" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=5"


# ---------- Response Fields ----------
echo "=== TC-036 image_ref echoed back ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=en" \
  -F "image_ref=img-ref-test-001" \
  -F "page=1" \
  -F "page_size=5"

echo "=== TC-038 Highest price product — AeroDry SAR 2500 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"sku": "qa-product-01"}' \
  -F "page=1" \
  -F "page_size=5"

echo "=== TC-039 Lowest price product — Rubber Band SAR 3.5 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"sku": "ao-100016_1"}' \
  -F "page=1" \
  -F "page_size=5"

echo "=== TC-040 Out of stock browse ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F 'filters={"availability": false}' \
  -F "page=1" \
  -F "page_size=10"


# ---------- Suggestions ----------
echo "=== TC-026 Suggestions — pres prefix ==="
curl -s "${base_url}/v1/search/suggestions?q=pres&language=en&product_limit=5&category_limit=3"

echo "=== TC-027 Suggestions — pump prefix ==="
curl -s "${base_url}/v1/search/suggestions?q=pump&language=en&product_limit=5&category_limit=3"

echo "=== TC-028 Suggestions — query_limit overrides product_limit ==="
curl -s "${base_url}/v1/search/suggestions?q=valve&language=en&query_limit=10&category_limit=3"

echo "=== TC-029 Suggestions — no match ==="
curl -s "${base_url}/v1/search/suggestions?q=zzzzzzz&language=en&product_limit=5&category_limit=3"


# ---------- Error Validation ----------
echo "=== TC-030 Invalid language — fr (422) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=fr"

echo "=== TC-031 Empty request — no query no filters (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en"

echo "=== TC-032 Invalid filter key (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=en" \
  -F 'filters={"unknown_key": "val"}'

echo "=== TC-033 page_size > 200 (422) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=en" \
  -F "page_size=500"

echo "=== TC-034 page < 1 (422) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "search_query=pipe" \
  -F "language=en" \
  -F "page=0"

echo "=== TC-035 Missing q param on suggestions (422) ==="
curl -s "${base_url}/v1/search/suggestions?language=en&product_limit=5"


# ---------- Diagnostics (new) ----------
echo "=== OpenSearch latency ==="
curl -s "${base_url}/health/latency/opensearch?samples=5"

echo "=== OCI latency ==="
curl -s "${base_url}/health/latency/oci?samples=5"

echo "=== Image search (expect 501) ==="
curl -s -X POST "${base_url}/v1/search" -F "channel=web" -F "language=en" -F "image=@/path/to/photo.bin;type=application/octet-stream"
