#!/usr/bin/env bash
# curl versions of every request in ASMO_Search_API.postman_collection.json
# (generated from the collection, same names and order).
#
# Run from this folder so the test_images*/ paths resolve:
#   ./curls.sh                                  # against http://localhost:8000
#   base_url=https://my-host ./curls.sh         # against another server
# On Windows use Git Bash or WSL.
#
# Image uploads: binary in a normal multipart image part — a file part with
# Content-Type image/jpeg, image/png or image/webp, max 10 MiB. A text field (e.g.
# Base64), any other Content-Type, GIF or non-image bytes get 400 UNSUPPORTED_IMAGE_TYPE
# ("Unsupported image type. Only JPEG, PNG and WebP images are accepted."); larger files 413.
# test_images_known/ lives in the data-pipeline repo; copy it next to this file.
base_url="${base_url:-http://localhost:8000}"
category_id="${category_id:-1ba050d2-f900-4a9b-82c0-9dade7948d72}"
# Filled in by TC-050 / TC-053 below (Redis embedding cache).
text_search_request_id="${text_search_request_id:-}"
image_search_request_id="${image_search_request_id:-}"

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
  -F "image=@test_images/ao-100018_1.jpg;type=image/jpeg"
echo

echo "=== TC-042 Image-only search — PNG (HPE X242 40G QSFP) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/a-880000629771.png;type=image/png"
echo

echo "=== TC-044 Text + image search (Air Handling Unit) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=air handling unit" \
  -F "image=@test_images/172000-001-hvac.png;type=image/png"
echo

echo "=== TC-045 Image + filters (material_type 9COM) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "filters={\"material_type\": [\"9COM\"], \"availability\": [\"IN_STOCK\"]}" \
  -F "image=@test_images/a-880000629771.png;type=image/png"
echo

echo "=== TC-046 Image search sorted by price asc ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "image=@test_images/ao-100018_1.jpg;type=image/jpeg"
echo

echo "=== TC-048 Generic binary content type (application/octet-stream) rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.jpg;type=application/octet-stream"
echo


# Image Search - Known Answers

echo "=== TC-IMG-01 Air Handling Unit 5 TR - 172000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-001-hvac__img0.png;type=image/png"
echo

echo "=== TC-IMG-02 Cooling Tower 3 TR - 172000-003-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-003-hvac__img0.png;type=image/png"
echo

echo "=== TC-IMG-03 Cooling Tower 3 TR - 174000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/174000-001-hvac__img0.jpg;type=image/jpeg"
echo

echo "=== TC-IMG-04 HPE X242 40G QSFP - a-880000629771 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629771__img0.png;type=image/png"
echo

echo "=== TC-IMG-05 5412R 92GT PoE+-4SFP+ v3 zl2 Swch - a-880000629773 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629773__img1.png;type=image/png"
echo

echo "=== TC-IMG-06 Stapler Standard Duty Black - ao-100004_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100004_1__img1.png;type=image/png"
echo

echo "=== TC-IMG-07 Scissors Stainless Steel 8 Inch - ao-100014_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100014_1__img0.png;type=image/png"
echo

echo "=== TC-IMG-08 Charger Battery (Uk PIN) 2100MAH AA - ao-100018_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100018_1__img0.jpg;type=image/jpeg"
echo

echo "=== TC-IMG-09 Calculator Desktop 12 Digit - ao-100021_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100021_1__img0.png;type=image/png"
echo

echo "=== TC-IMG-10 7X50 A211 ACULON BINO NIKON - ao-100027_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100027_1__img0.png;type=image/png"
echo

echo "=== TC-IMG-11 Book 20 Pages Sz Letter 80GSM - ao-100031_1 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100031_1__img2.png;type=image/png"
echo

echo "=== TC-IMG-12 AeroDry Smart Dehumidifier 20L - qa-product-01 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/qa-product-01__img3.png;type=image/png"
echo

echo "=== TC-IMG-13 Control Valve Positioner HART - 105 SKUs ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/shared__105_skus.jpg;type=image/jpeg"
echo

echo "=== TC-IMG-14 Text + image - extra unmatched word still finds 172000-001-hvac ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "search_query=air handling unit zzqx" \
  -F "image=@test_images_known/172000-001-hvac__img0.png;type=image/png"
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

echo "=== TC-043 Non-image file rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/not_an_image.png;type=image/png"
echo

echo "=== TC-043b GIF image rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/sample.gif;type=image/gif"
echo

echo "=== TC-043c Image sent as a text field (Base64) rejected (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=$(base64 < test_images/ao-100018_1.jpg | tr -d '\n')"
echo

echo "=== TC-047 Image larger than 10 MiB (413) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/oversized.jpg;type=image/jpeg"
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

echo "=== TC-050 Text search returns request_id ==="
resp=$(curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=gate valve")
echo "$resp"
text_search_request_id=$(printf '%s' "$resp" | _json_field request_id)
echo "-> text_search_request_id=${text_search_request_id}"
echo

echo "=== TC-051 Refine by request_id - sort by price, same request_id ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${text_search_request_id}" \
  -F "sort_by=price" \
  -F "sort_order=asc"
echo

echo "=== TC-052 Refine by request_id only - replays the query ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${text_search_request_id}"
echo

echo "=== TC-053 Image search returns request_id ==="
resp=$(curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.jpg;type=image/jpeg")
echo "$resp"
image_search_request_id=$(printf '%s' "$resp" | _json_field request_id)
echo "-> image_search_request_id=${image_search_request_id}"
echo

echo "=== TC-054 Refine image search by request_id - page 2 ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=2" \
  -F "page_size=10" \
  -F "request_id=${image_search_request_id}"
echo

echo "=== TC-055 Refine by request_id only - replays the image ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${image_search_request_id}"
echo

echo "=== TC-056 request_id + new search_query - text + image, keeps the image ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${image_search_request_id}" \
  -F "search_query=battery charger"
echo

echo "=== TC-057 Unknown / expired request_id ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=00000000-0000-0000-0000-000000000000"
echo

echo "=== TC-058 Malformed request_id (400) ==="
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=not-an-id"
echo
