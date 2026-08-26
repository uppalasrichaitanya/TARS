"""Small lexical similarity for memory retrieval without an embedding service.

TF-IDF cosine similarity is enough to find entries that share task vocabulary
across domain partitions while preserving the project's zero-dependency rule.
Its accepted limitation is semantic: pure paraphrases with no shared words do
not match. True embeddings would improve that case but require a dependency or
external API that memory deliberately does not have.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase and split text wherever characters are not alphanumeric."""
    return tuple(_TOKEN.findall(text.lower()))


def cosine_similarities(query: str, documents: Sequence[str]) -> tuple[float, ...]:
    """Score each document against `query` using corpus-local TF-IDF vectors."""
    if not documents:
        return ()

    tokenized = [tokenize(document) for document in documents]
    query_tokens = tokenize(query)
    if not query_tokens:
        return (0.0,) * len(documents)

    corpus = [*tokenized, query_tokens]
    document_frequency = Counter(token for tokens in corpus for token in set(tokens))
    corpus_size = len(corpus)
    inverse_frequency = {
        token: math.log((corpus_size + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }

    query_vector = _vector(query_tokens, inverse_frequency)
    return tuple(
        _cosine(query_vector, _vector(tokens, inverse_frequency))
        for tokens in tokenized
    )


def _vector(
    tokens: Sequence[str], inverse_frequency: dict[str, float]
) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {
        token: count / total * inverse_frequency[token]
        for token, count in counts.items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = left.keys() & right.keys()
    if not shared:
        return 0.0
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    return numerator / (left_norm * right_norm)
