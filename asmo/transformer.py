from .products import Product
from .prices import Price
from .logging_service import logger, log_function


@log_function
async def transform(products: list[Product], prices: list[Price], channel_id: str) -> dict[str, dict]:
    """Return a dict of sku → parsed row, one entry per SKU (masterVariant + all variants)."""
    logger.info(
        f"Transforming {len(products)} products and {len(prices)} prices",
        extra={"function_name": "transform", "input_data": {"products": len(products), "prices": len(prices)}},
    )
    prices_by_sku: dict[str, list[Price]] = {}
    for p in prices:
        prices_by_sku.setdefault(p.sku, []).append(p)

    logger.info(
        f"Price index built — {len(prices_by_sku)} unique SKUs across {len(prices)} prices",
        extra={"function_name": "transform", "input_data": {"unique_skus": len(prices_by_sku)}},
    )

    result = {}
    no_price: list[str] = []
    for product in products:
        rows = await _parse_product(product.raw, prices_by_sku, channel_id)
        for sku, row in rows.items():
            result[sku] = row
            if row.get("price") is None:
                no_price.append(sku)

    with_price = len(result) - len(no_price)
    coverage_pct = 100 * with_price / len(result) if result else 0
    logger.info(
        f"Transform complete — {len(result)} SKU rows | price coverage: {with_price}/{len(result)} ({coverage_pct:.1f}%)",
        extra={
            "function_name": "transform",
            "status": "success",
            "input_data": {
                "output_rows": len(result),
                "with_price": with_price,
                "without_price": len(no_price),
                "price_coverage_pct": round(coverage_pct, 1),
            },
        },
    )
    if no_price:
        logger.warning(
            f"{len(no_price)} SKU(s) have no matched price",
            extra={"function_name": "transform", "input_data": {"skus": no_price[:20]}},
        )
    return result


async def _parse_product(raw: dict, prices_by_sku: dict[str, list[Price]], channel_id: str) -> dict[str, dict]:
    try:
        product_fields = {
            **await _category_fields(raw),
            "productKey": raw.get("key"),
            "productName_en": raw.get("name", {}).get("en"),
            "productName_de": raw.get("name", {}).get("de"),
            "productDescription_en": raw.get("description", {}).get("en"),
            "productDescription_de": raw.get("description", {}).get("de"),
            "productSlug": raw.get("slug", {}).get("en"),
            "productGroup": await _attr(raw.get("attributes", []), "productGroup"),
            "materialType": await _attr(raw.get("attributes", []), "materialType"),
            "customerId": await _attr(raw.get("attributes", []), "ZCustId"),
            "customerMaterialNumber": await _attr(raw.get("attributes", []), "zCustMAT_9M"),
            "chemicalHazardBulletin": await _attr(raw.get("attributes", []), "chemicalHazardBulletin"),
            "hazardHealth": await _hazard(raw.get("attributes", []), "hazardRatingHealthChar", "hazardRatingHealthValue"),
            "hazardFlame": await _hazard(raw.get("attributes", []), "hazardRatingFlameChar", "hazardRatingFlameValue"),
            "hazardReactive": await _hazard(raw.get("attributes", []), "hazardRatingReactiveChar", "hazardRatingReactiveValue"),
        }

        logger.info(
            f"Extracted product-level fields for key={raw.get('key')!r} ({len(product_fields)} fields)",
            extra={
                "function_name": "_parse_product",
                "input_data": {"id": raw.get("id"), "key": raw.get("key"), "variants": 1 + len(raw.get("variants", []))},
                "response_payload": product_fields,
            },
        )

        all_variants = [raw.get("masterVariant", {}), *raw.get("variants", [])]

        total_qty = 0
        for v in all_variants:
            avail = await _channel_avail(v, channel_id)
            total_qty += avail.get("availableQuantity") or 0

        in_stock_any = False
        for v in all_variants:
            for ch in v.get("availability", {}).get("channels", {}).values():
                if ch.get("isOnStock"):
                    in_stock_any = True
                    break

        rows: dict[str, dict] = {}
        for v in all_variants:
            row = await _parse_variant(product_fields, v, prices_by_sku, channel_id, total_qty, in_stock_any)
            sku = row.get("sku")
            if sku:
                rows[sku] = row
        return rows
    except Exception as exc:
        logger.error(
            f"Failed to parse product key={raw.get('key')!r}: {exc}",
            extra={
                "function_name": "_parse_product",
                "status": "error",
                "error_details": str(exc),
                "input_data": {"id": raw.get("id"), "key": raw.get("key")},
            },
        )
        raise


async def _parse_variant(
    product_fields: dict,
    variant: dict,
    prices_by_sku: dict[str, list[Price]],
    channel_id: str,
    total_qty: int,
    in_stock_any: bool,
) -> dict:
    try:
        sku = variant.get("sku")
        avail = await _channel_avail(variant, channel_id)
        v_attrs = variant.get("attributes", [])

        sku_prices = prices_by_sku.get(sku, []) if sku else []
        contracted = [p for p in sku_prices if p.scope == "CONTRACTED"]
        pool = contracted if contracted else sku_prices
        min_price = min(pool, key=lambda p: p.amount, default=None)

        logger.info(
            f"SKU={sku!r} → total_prices={len(sku_prices)} contracted={len(contracted)} "
            f"selected={'CONTRACTED' if contracted else 'pool'} "
            f"price={f'{min_price.amount} {min_price.currency}' if min_price else 'none'} "
            f"qty={total_qty} in_stock={in_stock_any}",
            extra={
                "request_id": sku,
                "function_name": "_parse_variant",
                "input_data": {
                    "sku": sku,
                    "total_prices": len(sku_prices),
                    "contracted_prices": len(contracted),
                    "selected_price": min_price.amount if min_price else None,
                    "currency": min_price.currency if min_price else None,
                    "total_qty": total_qty,
                    "in_stock": in_stock_any,
                },
            },
        )

        row = {
            **product_fields,
            "sku": sku,
            "variantId": variant.get("id"),
            "variantKey": variant.get("key"),
            "manufacturerNumber": await _attr(v_attrs, "manufacturerNumber"),
            "productManufacturerNumber": await _attr(v_attrs, "productManufacturerNumber"),
            "erpMaterialNumber": await _attr(v_attrs, "ownInventoryManagedProduct"),
            "imageUrl": await _image_url(variant),
            "availableQuantity": avail.get("availableQuantity"),
            "restockableInDays": avail.get("restockableInDays"),
            "inStockAnyChannel": in_stock_any,
            "totalAvailableQuantity": total_qty,
            "currencyType": min_price.currency if min_price else None,
            "price": min_price.amount if min_price else None,
            "lowestContractedPrice": min_price.amount if min_price else None,
            "sellerName_en": min_price.seller_name_en if min_price else None,
            "sellerName_de": min_price.seller_name_de if min_price else None,
            "contractIds": [p.contract_id for p in contracted if p.contract_id] or None,
            "leadTimeDays": min_price.lead_time_days if min_price else None,
            "unitOfMeasure": next((p.price_unit for p in pool if p.price_unit is not None), None),
            "moq": next((p.moq for p in pool if p.moq is not None), None),
        }
        logger.info(
            f"Parsed row for sku={sku!r} ({len(row)} fields)",
            extra={
                "request_id": sku,
                "function_name": "_parse_variant",
                "input_data": {"sku": sku, "variantId": variant.get("id")},
                "response_payload": row,
            },
        )
        return row
    except Exception as exc:
        logger.error(
            f"Failed to parse variant sku={variant.get('sku')!r}: {exc}",
            extra={
                "request_id": variant.get("sku"),
                "function_name": "_parse_variant",
                "status": "error",
                "error_details": str(exc),
                "input_data": {"sku": variant.get("sku"), "variantId": variant.get("id")},
            },
        )
        raise


async def _category_fields(raw: dict) -> dict:
    categories = raw.get("categories", [])
    if not categories:
        return {
            "rootCategoryName": None, "rootCategoryKey": None,
            "rootCategorySlug": None, "rootCategoryOrderHint": None,
            "subCategoryName": None, "subCategoryKey": None,
            "subCategorySlug": None, "subCategoryDescription": None,
        }

    root_seen: set[str] = set()
    sub_seen: set[str] = set()
    roots: list[dict] = []
    subs: list[dict] = []

    for cat in categories:
        sub = cat.get("obj", {})
        sub_key = sub.get("key")

        # Collect subcategory (deduplicated by key)
        if sub_key and sub_key not in sub_seen:
            sub_seen.add(sub_key)
            subs.append(sub)

        # Skip CT internal "All" root (obj.ancestors is empty); first with ancestors is business root
        ancestors = sub.get("ancestors", [])
        business_ancestors = [a for a in ancestors if a.get("obj", {}).get("ancestors")]
        if business_ancestors:
            root = business_ancestors[0].get("obj", {})
            root_key = root.get("key")
            if root_key and root_key not in root_seen:
                root_seen.add(root_key)
                roots.append(root)

    def _pick(items: list[dict], field: str, nested: str | None = None):
        values = [
            (item.get(field, {}).get(nested) if nested else item.get(field))
            for item in items
        ]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    return {
        "rootCategoryName": _pick(roots, "name", "en"),
        "rootCategoryKey": _pick(roots, "key"),
        "rootCategorySlug": _pick(roots, "slug", "en"),
        "rootCategoryOrderHint": await _to_float(_pick(roots, "orderHint")),
        "subCategoryName": _pick(subs, "name", "en"),
        "subCategoryKey": _pick(subs, "key"),
        "subCategorySlug": _pick(subs, "slug", "en"),
        "subCategoryDescription": _pick([{**s, "desc": (s.get("description") or {}).get("en")} for s in subs], "desc"),
    }


async def _attr(attributes: list, name: str):
    for a in attributes or []:
        if a.get("name") == name:
            return a.get("value")
    return None


async def _hazard(attrs: list, char_key: str, value_key: str) -> dict | None:
    char = await _attr(attrs, char_key)
    val = await _attr(attrs, value_key)
    if char is None and val is None:
        return None
    return {"code": char, "value": val}


async def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


async def _image_url(variant: dict) -> str | None:
    images = variant.get("images", [])
    if images:
        return images[0].get("url")
    for asset in variant.get("assets", []):
        if "image" in asset.get("tags", []):
            sources = asset.get("sources", [])
            if sources:
                return sources[0].get("uri")
    return None


async def _channel_avail(variant: dict, channel_id: str) -> dict:
    return variant.get("availability", {}).get("channels", {}).get(channel_id, {})
