"""How much does each query word carry the query's meaning? (calibration for query relaxation)

For each query, embeds the full query and the query with each word left out, and prints how far
each leave-one-out version moves from the full query:

    drop     = 1 - cosine(full query, query without the word)   (0 = the word adds nothing)
    relative = drop / the largest drop in that query            (1.0 = the most important word)

Uses the app's own embedder (same model, settings and .env as the API), no OpenSearch.

    python diagnose_word_importance.py                      # the default queries below
    python diagnose_word_importance.py "I want lamps" "laptop for work"
"""
from __future__ import annotations

import asyncio
import math
import sys

from services.embedder import embedding_service

DEFAULT_QUERIES = [
    "I want lamps",                        # should keep: lamps
    "I want lamps for my studey table",    # should keep: lamps (table / studey are in no product)
    "laptop for work",                     # should NOT keep "work" (returned steel tubes before)
    "I want steel pipes",                  # should keep: steel, pipes
    "i want steel valvs",                  # should keep: steel, valvs
    "glass tube for cooking",
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _measure(query: str) -> None:
    words = query.split()
    if len(words) < 2:
        print(f"\n{query!r}: one word, nothing to compare")
        return
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    vectors = await asyncio.gather(embedding_service.encode_query(query),
                                   *(embedding_service.encode_query(v) for v in variants))
    full, rest = vectors[0], vectors[1:]
    drops = [1 - _cosine(full, v) for v in rest]
    top = max(drops) or 1.0
    print(f"\n{query!r}")
    print(f"  {'word':<14}{'drop':>8}{'relative':>10}   query without it")
    for word, drop, variant in sorted(zip(words, drops, variants), key=lambda t: -t[1]):
        print(f"  {word:<14}{drop:>8.4f}{drop / top:>10.2f}   {variant!r}")


async def main() -> None:
    queries = sys.argv[1:] or DEFAULT_QUERIES
    await embedding_service.load()
    for query in queries:
        await _measure(query)


if __name__ == "__main__":
    asyncio.run(main())
