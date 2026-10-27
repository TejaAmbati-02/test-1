# ASMO AI Search API — Test Cases (curl)

Every request in the **Bruno** (`ASMO_Search_API.bruno_collection.json`) and **Postman** (`ASMO_Search_API.postman_collection.json`) collections, as curl commands, with the same **TC IDs** and order. **Part A** lists the 50 cases that succeed; **Part B** lists the 23 error-path cases: 19 that must be rejected, with the exact error returned, and 4 upload variants (TC-043b–e) that are accepted.

## How to run

```bash
export base_url=http://localhost:8000          # or the deployed API URL
export category_id=1ba050d2-f900-4a9b-82c0-9dade7948d72
```

- Run from the API repo folder: image paths are relative to it (`test_images/`, `test_images_known/`).
- Commands are bash. On Windows use Git Bash, or `curl.exe` in PowerShell / cmd with each command on one line.
- `text_search_request_id` / `image_search_request_id` are the `request_id` returned by TC-050 / TC-053 (Part A, Redis section).
- The same commands are in `curls.sh`, which runs them all in order.

## Request rules

### `POST /v1/search` — `multipart/form-data`

| Field | Required | Values |
|---|---|---|
| `channel` | yes | `web` or `mobile` |
| `language` | yes | `en` (English only) |
| `search_query` | no | free text |
| `image` | no | binary multipart part — any file name, sent as `image/jpeg`, `image/png`, `image/webp` or `application/octet-stream`; max 10 MiB; the bytes must be a JPEG, PNG or WebP image (GIF is rejected) |
| `filters` | no | JSON object as a string, e.g. `{"price": {"gte": 50, "lte": 150}, "material_type": ["9CAT"]}` |
| `sort_by` | no | `relevance` (default) or `price` |
| `sort_order` | no | `desc` (default) or `asc` |
| `page` | no | 1-based, default `1` |
| `page_size` | no | 1–200, default `60` |
| `request_id` | no | `request_id` of a previous response; refines that search — its query and/or image are reused for whichever of `search_query` / `image` is omitted |

At least one of `search_query`, `image`, `filters` or `request_id` is required.

**request_id (search context)** — every response's `request_id` (a UUID) identifies its search: the query and/or image embeddings are stored as **one** Redis entry under it (the image itself is not stored; the entry expires 1 hour after it was last used). Send it back as `request_id` with only `filters` / `sort_by` / `sort_order` / `page` / `page_size` to refine the same search; the response keeps the same `request_id`. A `search_query` or `image` sent with it replaces that part — e.g. a new query over the previously uploaded image — which is a new search, so the response carries a new `request_id`.

#### Filter keys

Every facet key in the response is also a filter key: send back the facet's `id`. A value may be a single value or a list (matches any); numeric keys also take a range object (`gt` / `gte` / `lt` / `lte`). Any other key returns `400 INVALID_FILTER_KEY`.

| Filter key | Takes | Matches |
|---|---|---|
| `category_name` | category id (facet `id`) | `category_ids` |
| `sub_category_name` | sub-category id (facet `id`) | `sub_category_id` |
| `product_type` | product-type id (facet `id`); the product-type name is also accepted | `product_type_id` / `product_type` |
| `sellername` | seller id (facet `id`) | any `seller_details[].seller_id` |
| `moq` | number or range | primary seller's `moq` |
| `lead_time` | days, number or range | any `seller_details[].lead_time` |
| `material_type` | value, e.g. `9CAT` (facet `id`) | `material_type` |
| `price` | number or range, e.g. `{"gte": 50, "lte": 150}` (bounds from the `price` facet) | primary seller's `price` |
| `availability` | value, e.g. `IN_STOCK` | `availability` |
| `category_ids` | category id (same as `category_name`) | `category_ids` |
| `currency` | ISO 4217 code | primary seller's `currency` |
| `contract_indicator` | `true` / `false` | `contract_indicator` |
| `temporary_unavailable` | `true` / `false` | primary seller's `temporary_unavailable` |
| `supplier_count` | number or range | `supplier_count` |

Also accepted, by exact value: `sub_category_id`, `product_type_id`, `sold_by`, `sku_id`, `product_key`, `asmo_material_number`, `aramco_material_number`.

**Sending the image from the browser** — append the file as a normal part:

```js
const form = new FormData();
form.append('channel', 'web');
form.append('language', 'en');
form.append('image', file);  // a File from <input type="file">, or a Blob with its image type
await fetch(`${baseUrl}/v1/search`, { method: 'POST', body: form });  // do not set Content-Type yourself
```

### `GET /v1/search/suggestions`

Query parameters: `q` (required), `channel` (required), `language` (required), `query_limit` (1–50, default 5), `product_limit` (1–50, default 5).

## Responses

### Search — `200`

```json
{
  "request_id": "5b0c1f0e-\u2026",
  "total_results": 1,
  "page": 1,
  "page_size": 60,
  "products": [
    {
      "sku_id": "172000-001-hvac",
      "sku_name": "Air Handling Unit 5 TR",
      "primary_image_url": "https://\u2026",
      "asmo_material_number": "\u2026",
      "aramco_material_number": "\u2026",
      "seller_details": [
        {
          "seller_id": "3a7b23e0-\u2026",
          "name": "seller_123",
          "lead_time": 3,
          "unit_type": "Ea",
          "price": 123.0,
          "moq": 5,
          "currency": "SAR",
          "temporary_unavailable": null
        }
      ],
      "temporary_unavailable": null
    }
  ],
  "facets": {
    "category_name": [
      {
        "id": "1ba050d2-f900-4a9b-82c0-9dade7948d72",
        "value": "Drilling & Chemicals",
        "count": 1
      }
    ],
    "sub_category_name": [
      {
        "id": "13b62da1-d0f8-4208-8ff4-2f18df36fac4",
        "value": "Drilling",
        "count": 1
      }
    ],
    "product_type": [
      {
        "id": "0aca63c8-d90d-4dc0-b3a8-17b318cb2865",
        "value": "\u2026",
        "count": 1
      }
    ],
    "sellername": [
      {
        "id": "3a7b23e0-0047-4d63-88a9-cd05d1006a39",
        "value": "seller_123",
        "count": 1
      }
    ],
    "material_type": [
      {
        "id": "9COM",
        "value": "9COM",
        "count": 1
      }
    ],
    "price": {
      "min": 123.0,
      "max": 123.0
    }
  }
}
```

### Suggestions — `200`

```json
{
  "query": "pres",
  "query_suggestions": [
    {
      "text": "Pressure Transmitter HART"
    }
  ],
  "product_suggestions": [
    {
      "text": "Pressure Transmitter HART",
      "sku_id": "172000-001-hvac"
    }
  ]
}
```

### Errors

Every error has the same body: `{"code": "…", "message": "…"}`.

| HTTP | `code` | When |
|---|---|---|
| 400 | `UNSUPPORTED_CHANNEL` | `channel` is not `web` / `mobile` |
| 400 | `UNSUPPORTED_LANGUAGE` | `language` is not `en` |
| 400 | `INVALID_SORT_BY` | `sort_by` is not `relevance` / `price` |
| 400 | `EMPTY_REQUEST` | no `search_query`, `image`, `filters` or `request_id` |
| 400 | `INVALID_FILTER` | `filters` is not a valid JSON object |
| 400 | `INVALID_FILTER_KEY` | a filter key is not supported |
| 400 | `INVALID_FILTER_VALUE` | a filter value has the wrong type |
| 400 | `VALIDATION_ERROR` | a field is missing or out of range (e.g. `page_size` > 200, `page` < 1) |
| 400 | `UNSUPPORTED_IMAGE_TYPE` | the part's content type is not JPEG / PNG / WebP / `application/octet-stream`, or its bytes are not a JPEG, PNG or WebP image — GIF included (`Unsupported image. Upload a JPEG, PNG or WebP image.`) |
| 413 | `IMAGE_TOO_LARGE` | image larger than 10 MiB |
| 400 | `REQUEST_ID_NOT_FOUND` | `request_id` is not a UUID, unknown, or its search has expired |
| 503 | `EMBEDDING_STORE_UNAVAILABLE` | a `request_id` was sent but the cache (Redis) is unavailable |
| 404 | `INDEX_NOT_FOUND` | the search index does not exist |
| 500 | `INTERNAL_ERROR` | unexpected server error |

## Part A — Pass cases

### Health & Infra

#### A1. TC-000 Health Check

`GET /health` → expected **200**

```bash
curl -s "${base_url}/health"
```

#### A2. TC-000b List Indices

`GET /indices` → expected **200**

Checks: indices is an array.

```bash
curl -s "${base_url}/indices"
```

### Text Search

#### A3. TC-001 Search — pipe (en)

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pipe" \
  -F "sort_by=relevance" \
  -F "sort_order=desc" \
  -F "page=1" \
  -F "page_size=60"
```

#### A4. TC-002 Search — pump (en)

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
```

#### A5. TC-003 Search — valve (mobile)

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=mobile" \
  -F "language=en" \
  -F "search_query=valve" \
  -F "page=1" \
  -F "page_size=60"
```

### Sort

#### A6. TC-009 Sort by price asc

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; prices non-decreasing; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "page=1" \
  -F "page_size=60"
```

#### A7. TC-010 Sort by price desc

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; prices non-increasing; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=price" \
  -F "sort_order=desc" \
  -F "page=1" \
  -F "page_size=60"
```

### Filter Browse

#### A8. TC-011 Browse — in stock only

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"availability\": [\"IN_STOCK\"]}" \
  -F "page=1" \
  -F "page_size=60"
```

#### A9. TC-014 Browse — price range 50–150

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; prices within 50..150; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"price\": {\"gte\": 50, \"lte\": 150}}" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "page=1" \
  -F "page_size=60"
```

#### A10. TC-016 Browse — material type 9CAT

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields; material_type facet is only 9CAT.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"material_type\": [\"9CAT\"]}" \
  -F "page=1" \
  -F "page_size=60"
```

#### A11. TC-012 Browse — category_ids

Category browse: same endpoint, no search_query, category passed inside filters.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields; category facet is only the category.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"category_ids\": [\"${category_id}\"]}" \
  -F "page=1" \
  -F "page_size=60"
```

### Combined Search + Filters

#### A12. TC-018 Search + availability filter

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"availability\": [\"IN_STOCK\"]}" \
  -F "page=1" \
  -F "page_size=60"
```

#### A13. TC-019 Search + price range filter

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"price\": {\"lte\": 500}}" \
  -F "page=1" \
  -F "page_size=60"
```

#### A14. TC-020 Search + spec example filters (material_type, availability, price)

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is an array; page = 1; page_size = 60; request_id is a UUID; facets is a keyed object; no legacy fields; products expose spec fields; seller_details expose spec fields; filters honoured.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "filters={\"material_type\": [\"9CAT\"], \"availability\": [\"IN_STOCK\"], \"price\": {\"lte\": 500}}" \
  -F "page=1" \
  -F "page_size=60"
```

### Pagination

#### A15. TC-024 Pagination — page 1

`POST /v1/search` → expected **200**

Checks: page = 1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
```

#### A16. TC-025 Pagination — page 2 differs from page 1

`POST /v1/search` → expected **200**

Checks: page = 2; page 2 first differs from page 1 (when enough results).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=2" \
  -F "page_size=60"
```

### Image Search

#### A17. TC-041 Image-only search — JPEG (Charger, Battery)

Image search: binary JPEG upload, no search_query. Uses SKU ao-100018_1's own product image.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; ao-100018_1 is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
```

#### A18. TC-042 Image-only search — PNG (HPE X242 40G QSFP)

Image search: binary PNG upload, no search_query. Uses SKU a-880000629771's own product image.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; a-880000629771 is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/a-880000629771.bin;type=application/octet-stream"
```

#### A19. TC-044 Text + image search (Air Handling Unit)

Multimodal search: search_query and binary image together.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; 172000-001-hvac is in the top 3 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=air handling unit" \
  -F "image=@test_images/172000-001-hvac.bin;type=application/octet-stream"
```

#### A20. TC-045 Image + filters (material_type 9COM)

Image search combined with filters; every result must satisfy the filters.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; a-880000629771 is in the top 5 (searched with its own image); material_type facet is only 9COM.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "filters={\"material_type\": [\"9COM\"], \"availability\": [\"IN_STOCK\"]}" \
  -F "image=@test_images/a-880000629771.bin;type=application/octet-stream"
```

#### A21. TC-046 Image search sorted by price asc

Image search with sorting.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; prices non-decreasing.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "sort_by=price" \
  -F "sort_order=asc" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
```

#### A22. TC-048 Generic binary content type (application/octet-stream)

PNG bytes sent with Content-Type application/octet-stream; the API detects PNG from the bytes.

`POST /v1/search` → expected **200**

Checks: total_results is a number; products is a non-empty array; facets is a keyed object; products expose spec fields; no image vectors leak into products; 172000-001-hvac is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/172000-001-hvac.bin;type=application/octet-stream"
```

### Image Search - Known Answers

#### A23. TC-IMG-01 Air Handling Unit 5 TR - 172000-001-hvac

Known-answer image search: 172000-001-hvac__img0.png is a byte-identical copy of the product image ingested for 172000-001-hvac, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is 172000-001-hvac.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-001-hvac__img0.bin;type=application/octet-stream"
```

#### A24. TC-IMG-02 Cooling Tower 3 TR - 172000-003-hvac

Known-answer image search: 172000-003-hvac__img0.png is a byte-identical copy of the product image ingested for 172000-003-hvac, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is 172000-003-hvac.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/172000-003-hvac__img0.bin;type=application/octet-stream"
```

#### A25. TC-IMG-03 Cooling Tower 3 TR - 174000-001-hvac

Known-answer image search: 174000-001-hvac__img0.jpg is a byte-identical copy of the product image ingested for 174000-001-hvac, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is 174000-001-hvac.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/174000-001-hvac__img0.bin;type=application/octet-stream"
```

#### A26. TC-IMG-04 HPE X242 40G QSFP - a-880000629771

Known-answer image search: a-880000629771__img0.png is a byte-identical copy of the product image ingested for a-880000629771, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is a-880000629771.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629771__img0.bin;type=application/octet-stream"
```

#### A27. TC-IMG-05 5412R 92GT PoE+-4SFP+ v3 zl2 Swch - a-880000629773

Known-answer image search: a-880000629773__img1.png is a byte-identical copy of the product image ingested for a-880000629773, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is a-880000629773.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/a-880000629773__img1.bin;type=application/octet-stream"
```

#### A28. TC-IMG-06 Stapler Standard Duty Black - ao-100004_1

Known-answer image search: ao-100004_1__img1.png is a byte-identical copy of the product image ingested for ao-100004_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100004_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100004_1__img1.bin;type=application/octet-stream"
```

#### A29. TC-IMG-07 Scissors Stainless Steel 8 Inch - ao-100014_1

Known-answer image search: ao-100014_1__img0.png is a byte-identical copy of the product image ingested for ao-100014_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100014_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100014_1__img0.bin;type=application/octet-stream"
```

#### A30. TC-IMG-08 Charger Battery (Uk PIN) 2100MAH AA - ao-100018_1

Known-answer image search: ao-100018_1__img0.jpg is a byte-identical copy of the product image ingested for ao-100018_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100018_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100018_1__img0.bin;type=application/octet-stream"
```

#### A31. TC-IMG-09 Calculator Desktop 12 Digit - ao-100021_1

Known-answer image search: ao-100021_1__img0.png is a byte-identical copy of the product image ingested for ao-100021_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100021_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100021_1__img0.bin;type=application/octet-stream"
```

#### A32. TC-IMG-10 7X50 A211 ACULON BINO NIKON - ao-100027_1

Known-answer image search: ao-100027_1__img0.png is a byte-identical copy of the product image ingested for ao-100027_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100027_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100027_1__img0.bin;type=application/octet-stream"
```

#### A33. TC-IMG-11 Book 20 Pages Sz Letter 80GSM - ao-100031_1

Known-answer image search: ao-100031_1__img2.png is a byte-identical copy of the product image ingested for ao-100031_1, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is ao-100031_1.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/ao-100031_1__img2.bin;type=application/octet-stream"
```

#### A34. TC-IMG-12 AeroDry Smart Dehumidifier 20L - qa-product-01

Known-answer image search: qa-product-01__img3.png is a byte-identical copy of the product image ingested for qa-product-01, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is qa-product-01.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/qa-product-01__img3.bin;type=application/octet-stream"
```

#### A35. TC-IMG-13 Control Valve Positioner HART - 105 SKUs

Known-answer image search: shared__105_skus.jpg is a byte-identical copy of the product image ingested for one of the 105 SKUs sharing this image, so it must rank first.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; top hit is one of the 105 SKUs sharing this image; every product on page 1 shares this image.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images_known/shared__105_skus.bin;type=application/octet-stream"
```

#### A36. TC-IMG-14 Text + image - extra unmatched word still finds 172000-001-hvac

Text + image fusion: 'zzqx' appears in no product, which used to filter every result out (all words had to match). With weighted RRF the text only helps ranking, so the image match must still rank in the top 3. Fails with 0 products if the cluster rejected weighted RRF (API log: 'weighted RRF not available').

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; 172000-001-hvac is in the top 3.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "search_query=air handling unit zzqx" \
  -F "image=@test_images_known/172000-001-hvac__img0.bin;type=application/octet-stream"
```

### Suggestions

#### A37. TC-026 Suggestions — 'pres' prefix

`GET /v1/search/suggestions` → expected **200**

Checks: response keys are exactly the spec keys; query echoed; query_suggestions is array; product_suggestions is array; every query suggestion has only text; every product suggestion has only text; query_suggestions count <= 4; product_suggestions count <= 5.

```bash
curl -s "${base_url}/v1/search/suggestions?q=pres&channel=web&language=en&query_limit=4&product_limit=5"
```

#### A38. TC-027 Suggestions — 'pump' prefix (limit 3)

`GET /v1/search/suggestions` → expected **200**

Checks: product_suggestions count <= 3.

```bash
curl -s "${base_url}/v1/search/suggestions?q=pump&channel=web&language=en&query_limit=4&product_limit=3"
```

#### A39. TC-029 Suggestions — no match

`GET /v1/search/suggestions` → expected **200**

Checks: empty product suggestions; empty query suggestions.

```bash
curl -s "${base_url}/v1/search/suggestions?q=zzzzzzzz&channel=web&language=en&query_limit=4&product_limit=5"
```

### Diagnostics (operations only — not needed by the frontend)

#### A40. OpenSearch latency

`GET /health/latency/opensearch` → expected **200**

```bash
curl -s "${base_url}/health/latency/opensearch?samples=5"
```

#### A41. OCI latency

`GET /health/latency/oci` → expected **200**

```bash
curl -s "${base_url}/health/latency/oci?samples=5"
```

### Redis (Embedding Cache)

#### A42. Redis latency

Round-trip PING to Redis. With redis_enabled=false the API reports ok=false and that Redis is disabled.

`GET /health/latency/redis` → expected **200**

Checks: Redis reachable; reported as disabled.

```bash
curl -s "${base_url}/health/latency/redis?samples=5"
```

#### A43. TC-049 Health reports Redis status

redis_connected is true when Redis is enabled and reachable, null when disabled.

`GET /health` → expected **200**

Checks: redis_connected = true; redis_connected = null (disabled).

```bash
curl -s "${base_url}/health"
```

#### A44. TC-050 Text search returns request_id

First text search: its query embedding is stored in Redis under the response's request_id (saved as text_search_request_id).

`POST /v1/search` → expected **200**

Checks: request_id is a UUID.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=gate valve"
```

#### A45. TC-051 Refine by request_id - sort by price, same request_id

request_id plus sort_by / sort_order only: the same search (query reused, not re-sent), re-sorted. The response keeps the same request_id. Redis disabled → 503.

`POST /v1/search` → expected **200** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Checks: products is an array; same request_id (same search context); prices non-decreasing.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${text_search_request_id}" \
  -F "sort_by=price" \
  -F "sort_order=asc"
```

#### A46. TC-052 Refine by request_id only - replays the query

request_id alone reruns the same search from the stored query and embedding. Redis disabled → 503.

`POST /v1/search` → expected **200** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Checks: products is an array; same request_id (same search context).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${text_search_request_id}"
```

#### A47. TC-053 Image search returns request_id

The image embedding is stored in Redis under the response's request_id (saved as image_search_request_id); the image itself is not stored.

`POST /v1/search` → expected **200**

Checks: request_id is a UUID.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/ao-100018_1.bin;type=application/octet-stream"
```

#### A48. TC-054 Refine image search by request_id - page 2

request_id plus page: the next page of the same image search, without re-uploading the image. Redis disabled → 503.

`POST /v1/search` → expected **200** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Checks: products is an array; same request_id (same search context).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=2" \
  -F "page_size=10" \
  -F "request_id=${image_search_request_id}"
```

#### A49. TC-055 Refine by request_id only - replays the image

request_id alone reruns the image search from the stored embedding; no upload. Redis disabled → 503.

`POST /v1/search` → expected **200** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Checks: products is an array; same request_id (same search context); ao-100018_1 in top 5.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${image_search_request_id}"
```

#### A50. TC-056 request_id + new search_query - text + image, keeps the image

The image search's request_id plus a search_query: the stored image is reused (not re-uploaded) with the new query. That is a new search context, so the response has a new request_id. Redis disabled → 503.

`POST /v1/search` → expected **200** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Checks: products is an array; a new request_id for the new search context.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=${image_search_request_id}" \
  -F "search_query=battery charger"
```

## Part B — Fail cases (must be rejected)

### Error Validation

#### B1. TC-030 Unsupported language — fr (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_LANGUAGE","message":"Unsupported language 'fr'. Allowed: en"}
```

Checks: code = UNSUPPORTED_LANGUAGE; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=fr" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
```

#### B2. TC-008 Arabic language not supported (400)

Only English is supported (the index holds English text only): language=ar is rejected.

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_LANGUAGE","message":"Unsupported language 'ar'. Allowed: en"}
```

Checks: code = UNSUPPORTED_LANGUAGE; message lists only .

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=ar" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
```

#### B3. TC-030b Unsupported channel — desktop (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_CHANNEL","message":"Unsupported channel 'desktop'. Allowed: mobile, web"}
```

Checks: code = UNSUPPORTED_CHANNEL; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=desktop" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=60"
```

#### B4. TC-031 Empty request — no query/image/filters (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"EMPTY_REQUEST","message":"At least one of search_query, image, filters or request_id must be provided"}
```

Checks: code = EMPTY_REQUEST; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60"
```

#### B5. TC-032 Invalid filter key (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"INVALID_FILTER_KEY","message":"The filter key 'unknown_field' is not supported."}
```

Checks: code = INVALID_FILTER_KEY; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"unknown_field\": \"x\"}" \
  -F "page=1" \
  -F "page_size=60"
```

#### B6. TC-032b Malformed filters JSON (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"INVALID_FILTER","message":"filters must be a valid JSON object: Expecting property name enclosed in double quotes"}
```

Checks: code = INVALID_FILTER; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={not json}" \
  -F "page=1" \
  -F "page_size=60"
```

#### B7. TC-032c Invalid sort_by (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"INVALID_SORT_BY","message":"Unsupported sort_by 'name'. Allowed: price, relevance"}
```

Checks: code = INVALID_SORT_BY; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "sort_by=name" \
  -F "page=1" \
  -F "page_size=60"
```

#### B8. TC-033 page_size > 200 (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"VALIDATION_ERROR","message":"page_size: Input should be less than or equal to 200"}
```

Checks: code = VALIDATION_ERROR; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=1" \
  -F "page_size=500"
```

#### B9. TC-034 page < 1 (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"VALIDATION_ERROR","message":"page: Input should be greater than or equal to 1"}
```

Checks: code = VALIDATION_ERROR; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "search_query=pump" \
  -F "page=0" \
  -F "page_size=60"
```

#### B10. TC-035 Missing q on suggestions (400)

`GET /v1/search/suggestions` → expected **400**

Response:

```json
{"code":"VALIDATION_ERROR","message":"query.q: Field required"}
```

Checks: code = VALIDATION_ERROR; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s "${base_url}/v1/search/suggestions?channel=web&language=en"
```

#### B11. TC-035b Missing channel on suggestions (400)

`GET /v1/search/suggestions` → expected **400**

Response:

```json
{"code":"VALIDATION_ERROR","message":"query.channel: Field required"}
```

Checks: code = VALIDATION_ERROR; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s "${base_url}/v1/search/suggestions?q=pump&language=en"
```

#### B12. TC-035c Unsupported language on suggestions (400)

`GET /v1/search/suggestions` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_LANGUAGE","message":"Unsupported language 'fr'. Allowed: en"}
```

Checks: code = UNSUPPORTED_LANGUAGE; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s "${base_url}/v1/search/suggestions?q=pump&channel=web&language=fr"
```

#### B13. TC-043 Non-image file (400)

A file whose bytes are not a JPEG, PNG or WebP image is rejected by the byte check.

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_IMAGE_TYPE","message":"Unsupported image. Upload a JPEG, PNG or WebP image."}
```

Checks: code = UNSUPPORTED_IMAGE_TYPE; message is a non-empty string.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "search_query=pump" \
  -F "image=@test_images/not_an_image.bin;type=application/octet-stream"
```

#### B14. TC-043b JPEG image part (image/jpeg) accepted

A normal image part: a .jpg file sent as image/jpeg.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; ao-100018_1 is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/ao-100018_1.jpg;type=image/jpeg"
```

#### B15. TC-043c PNG image part (image/png) accepted

A normal image part: a .png file sent as image/png.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; 172000-001-hvac is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/172000-001-hvac.png;type=image/png"
```

#### B16. TC-043d File without an extension accepted

The file name does not matter; JPEG bytes with no extension are accepted.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; ao-100018_1 is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/image_no_extension;type=application/octet-stream"
```

#### B17. TC-043e Mislabelled content type accepted - the bytes decide

JPEG bytes declared as image/png: the API detects JPEG from the bytes.

`POST /v1/search` → expected **200**

Checks: products is a non-empty array; ao-100018_1 is in the top 5 (searched with its own image).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/ao-100018_1.bin;type=image/png"
```

#### B18. TC-043f GIF image rejected (400)

Only JPEG, PNG and WebP images are accepted; a GIF is rejected.

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_IMAGE_TYPE","message":"Unsupported image. Upload a JPEG, PNG or WebP image."}
```

Checks: code = UNSUPPORTED_IMAGE_TYPE; message = unsupported image.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/sample.gif;type=image/gif"
```

#### B19. TC-043g Unsupported content type rejected (400)

The image part must be sent as image/jpeg, image/png, image/webp or application/octet-stream; application/pdf is rejected.

`POST /v1/search` → expected **400**

Response:

```json
{"code":"UNSUPPORTED_IMAGE_TYPE","message":"Unsupported image. Upload a JPEG, PNG or WebP image."}
```

Checks: code = UNSUPPORTED_IMAGE_TYPE; message = unsupported image.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=10" \
  -F "image=@test_images/ao-100018_1.jpg;type=application/pdf"
```

#### B20. TC-047 Image larger than 10 MiB (413)

Uploads over 10 MiB (10485760 bytes) are rejected before embedding.

`POST /v1/search` → expected **413**

Response:

```json
{"code":"IMAGE_TOO_LARGE","message":"Image exceeds the maximum size of 10485760 bytes."}
```

Checks: code = IMAGE_TOO_LARGE; message is a non-empty string.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "image=@test_images/oversized.bin;type=application/octet-stream"
```

#### B21. TC-032d Invalid filter value (400)

`POST /v1/search` → expected **400**

Response:

```json
{"code":"INVALID_FILTER_VALUE","message":"Invalid value for filter 'price': \"abc\""}
```

Checks: code = INVALID_FILTER_VALUE; message is a non-empty string; flat error body (no wrapper).

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "filters={\"price\": \"abc\"}" \
  -F "page=1" \
  -F "page_size=60"
```

### Redis (Embedding Cache)

#### B22. TC-057 Unknown / expired request_id

A well-formed request_id whose search is not in Redis: 400 when Redis is enabled, 503 when disabled.

`POST /v1/search` → expected **400** (or **503** `EMBEDDING_STORE_UNAVAILABLE` when Redis is disabled)

Response:

```json
{"code":"REQUEST_ID_NOT_FOUND","message":"request_id '00000000-0000-0000-0000-000000000000' was not found or its search has expired"}
```

Checks: code = REQUEST_ID_NOT_FOUND; code = EMBEDDING_STORE_UNAVAILABLE.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=00000000-0000-0000-0000-000000000000"
```

#### B23. TC-058 Malformed request_id (400)

request_id must be a UUID; rejected without touching Redis (same with Redis on or off).

`POST /v1/search` → expected **400**

Response:

```json
{"code":"REQUEST_ID_NOT_FOUND","message":"'not-an-id' is not a valid request_id"}
```

Checks: code = REQUEST_ID_NOT_FOUND.

```bash
curl -s -X POST "${base_url}/v1/search" \
  -F "channel=web" \
  -F "language=en" \
  -F "page=1" \
  -F "page_size=60" \
  -F "request_id=not-an-id"
```
