#!/usr/bin/env bash
# curl versions of every request in ASMO_Search_API.postman_collection.json
# (generated from the collection, same names and order).
#
# Run from this folder so the test_images*/ paths resolve:
#   ./curls.sh                                  # against http://localhost:8000
#   base_url=https://my-host ./curls.sh         # against another server
# On Windows use Git Bash or WSL.
#
# Image uploads: only .bin files are accepted (file name ending in .bin, sent as
# application/octet-stream). Any other file gets 400 "Incorrect file format".
# test_images_known/ lives in the data-pipeline repo; copy it next to this file.
base_url="${base_url:-http://localhost:8000}"
category_id="${category_id:-1ba050d2-f900-4a9b-82c0-9dade7948d72}"
# Filled in by TC-050 / TC-053 below (Redis embedding cache).
query_embedding_id="${query_embedding_id:-}"
image_embedding_id="${image_embedding_id:-}"

PY="$(command -v python3 || command -v python)"
_json_field() {  # _json_field <field>  — read a top-level field from JSON on stdin
  "$PY" -c 'import sys, json; print(json.load(sys.stdin).get(sys.argv[1]) or "")' "$1" 2>/dev/null
}


# Health & Infra

echo "=== TC-000 Health Check ==="
curl -s "${base_url}/health"
echo

echo "=== TC-000b List Indices ==="
curl -s "${base_url}/indices"
echo


# Text Search

echo "=== TC-001 Search — pipe (en) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pipe" \
  -F "sort_by=relevance" \
  -F "sort_order=desc" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-002 Search — pump (en) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-003 Search — valve (mobile) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=mobile" \
  -F "language=en" \
  -F "search_query=valve" \
  -F "page=1" \
  -F "page_size=60"
echo


# Sort

echo "=== TC-009 Sort by price asc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-010 Sort by price desc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=price" \
  -F "sort_order=desc" \
  -F "page=1" \
  -F "page_size=60"
echo


# Filter Browse

echo "=== TC-011 Browse — in stock only ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"availability\": [\"IN_STOCK\"]}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-014 Browse — price range 50–150 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"price\": {\"gte\": 50, \"lte\": 150}}" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-016 Browse — material type 9CAT ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"material_type\": [\"9CAT\"]}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-012 Browse — category_ids ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"category_ids\": [\"${category_id}\"]}" \
  -F "page=1" \
  -F "page_size=60"
echo


# Combined Search + Filters

echo "=== TC-018 Search + availability filter ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"availability\": [\"IN_STOCK\"]}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-019 Search + price range filter ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"price\": {\"lte\": 500}}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-020 Search + spec example filters (material_type, availability, price) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"material_type\": [\"9CAT\"], \"availability\": [\"IN_STOCK\"], \"price\": {\"lte\": 500}}" \
  -F "page=1" \
  -F "page_size=60"
echo


# Pagination

echo "=== TC-024 Pagination — page 1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-025 Pagination — page 2 differs from page 1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=2" \
  -F "page_size=60"
echo


# Image Search

echo "=== TC-041 Image-only search — JPEG (Charger, Battery) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
echo

echo "=== TC-042 Image-only search — PNG (HPE X242 40G QSFP) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/a-880000629771.bin;type=application/octet-stream"
echo

echo "=== TC-044 Text + image search (Air Handling Unit) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=air handling unit" \
  -F "image=@test_images/172000-001-hvac.bin;type=application/octet-stream"
echo

echo "=== TC-045 Image + filters (material_type 9COM) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "filters={\"material_type\": [\"9COM\"], \"availability\": [\"IN_STOCK\"]}" \
  -F "image=@test_images/a-880000629771.bin;type=application/octet-stream"
echo

echo "=== TC-046 Image search sorted by price asc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
echo

echo "=== TC-048 Generic binary content type (application/octet-stream) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/172000-001-hvac.bin;type=application/octet-stream"
echo


# Image Search - Known Answers

echo "=== TC-IMG-01 Air Handling Unit 5 TR - 172000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-001-hvac__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-02 Cooling Tower 3 TR - 172000-003-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-003-hvac__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-03 Cooling Tower 3 TR - 174000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/174000-001-hvac__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-04 HPE X242 40G QSFP - a-880000629771 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629771__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-05 5412R 92GT PoE+-4SFP+ v3 zl2 Swch - a-880000629773 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629773__img1.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-06 Stapler Standard Duty Black - ao-100004_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100004_1__img1.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-07 Scissors Stainless Steel 8 Inch - ao-100014_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100014_1__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-08 Charger Battery (Uk PIN) 2100MAH AA - ao-100018_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100018_1__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-09 Calculator Desktop 12 Digit - ao-100021_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100021_1__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-10 7X50 A211 ACULON BINO NIKON - ao-100027_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100027_1__img0.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-11 Book 20 Pages Sz Letter 80GSM - ao-100031_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100031_1__img2.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-12 AeroDry Smart Dehumidifier 20L - qa-product-01 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/qa-product-01__img3.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-13 Control Valve Positioner HART - 105 SKUs ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/shared__105_skus.bin;type=application/octet-stream"
echo

echo "=== TC-IMG-14 Text + image - extra unmatched word still finds 172000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "search_query=air handling unit zzqx" \
  -F "image=@test_images_known/172000-001-hvac__img0.bin;type=application/octet-stream"
echo


# Suggestions

echo "=== TC-026 Suggestions — 'pres' prefix ==="
curl -s "${base_url}/v1/search/suggestions?q=pres&channel=web&language=en&query_limit=4&product_limit=5"
echo

echo "=== TC-027 Suggestions — 'pump' prefix (limit 3) ==="
curl -s "${base_url}/v1/search/suggestions?q=pump&channel=web&language=en&query_limit=4&product_limit=3"
echo

echo "=== TC-029 Suggestions — no match ==="
curl -s "${base_url}/v1/search/suggestions?q=zzzzzzzz&channel=web&language=en&query_limit=4&product_limit=5"
echo


# Error Validation

echo "=== TC-030 Unsupported language — fr (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=fr" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-008 Arabic language not supported (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=ar" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-030b Unsupported channel — desktop (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=desktop" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-031 Empty request — no query/image/filters (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-032 Invalid filter key (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"unknown_field\": \"x\"}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-032b Malformed filters JSON (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={not json}" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-032c Invalid sort_by (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=name" \
  -F "page=1" \
  -F "page_size=60"
echo

echo "=== TC-033 page_size > 200 (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=500"
echo

echo "=== TC-034 page < 1 (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=0" \
  -F "page_size=60"
echo

echo "=== TC-035 Missing q on suggestions (400) ==="
curl -s "${base_url}/v1/search/suggestions?channel=web&language=en"
echo

echo "=== TC-035b Missing channel on suggestions (400) ==="
curl -s "${base_url}/v1/search/suggestions?q=pump&language=en"
echo

echo "=== TC-035c Unsupported language on suggestions (400) ==="
curl -s "${base_url}/v1/search/suggestions?q=pump&channel=web&language=fr"
echo

echo "=== TC-043 Non-image file sent as bin (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=pump" \
  -F "image=@test_images/not_an_image.bin;type=application/octet-stream"
echo

echo "=== TC-043b JPG file rejected - only .bin accepted (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/ao-100018_1.jpg;type=image/jpeg"
echo

echo "=== TC-043c PNG file rejected - only .bin accepted (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/172000-001-hvac.png;type=image/png"
echo

echo "=== TC-043d File without .bin extension rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/image_no_extension;type=application/octet-stream"
echo

echo "=== TC-043e .bin file declared image-png rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/ao-100018_1.bin;type=image/png"
echo

echo "=== TC-047 Image larger than 5 MB (413) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/oversized.bin;type=application/octet-stream"
echo

echo "=== TC-032d Invalid filter value (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"price\": \"abc\"}" \
  -F "page=1" \
  -F "page_size=60"
echo


# Diagnostics

echo "=== OpenSearch latency ==="
curl -s "${base_url}/health/latency/opensearch?samples=5"
echo

echo "=== OCI latency ==="
curl -s "${base_url}/health/latency/oci?samples=5"
echo


# Redis (Embedding Cache)

echo "=== Redis latency ==="
curl -s "${base_url}/health/latency/redis?samples=5"
echo

echo "=== TC-049 Health reports Redis status ==="
curl -s "${base_url}/health"
echo

echo "=== TC-050 Text search returns query_embedding_id ==="
resp=$(curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=gate valve")
echo "$resp"
query_embedding_id=$(printf '%s' "$resp" | _json_field query_embedding_id)
echo "-> query_embedding_id=${query_embedding_id}"
echo

echo "=== TC-051 Same query again (different case/spaces) returns the same id ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=  Gate   VALVE "
echo

echo "=== TC-052 Search by query_embedding_id only (no search_query) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "query_embedding_id=${query_embedding_id}"
echo

echo "=== TC-053 Image search returns image_embedding_id ==="
resp=$(curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream")
echo "$resp"
image_embedding_id=$(printf '%s' "$resp" | _json_field image_embedding_id)
echo "-> image_embedding_id=${image_embedding_id}"
echo

echo "=== TC-054 Same image again returns the same id ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
echo

echo "=== TC-055 Search by image_embedding_id only (no image) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image_embedding_id=${image_embedding_id}"
echo

echo "=== TC-056 Text + image by both ids ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "query_embedding_id=${query_embedding_id}" \
  -F "image_embedding_id=${image_embedding_id}"
echo

echo "=== TC-057 Unknown / expired embedding id ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "query_embedding_id=asmo:emb:query:0000000000000000000000000000000000000000000000000000000000000000"
echo

echo "=== TC-058 Malformed embedding id (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image_embedding_id=not-an-id"
echo
