"""Dedup utilities for news collection, tagging, deduplication, and storage utilities."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import DedupHashes

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src", "source",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def canonicalize_url(url: str) -> str:
    """Canonicalize url for news collection, tagging, deduplication, and storage utilities.

    Args:
        url (str): The url value.

    Returns:
        str: The computed or requested result.
    """
    parts = urlsplit(url.strip())
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
             if k.lower() not in _TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, urlencode(sorted(query)), ""))


def _normalize_text(s: str) -> str:
    """Normalize text for news collection, tagging, deduplication, and storage utilities.

    Args:
        s (str): The s value.

    Returns:
        str: The computed or requested result.
    """
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def title_hash(title: str) -> str:
    """Title hash for news collection, tagging, deduplication, and storage utilities.

    Args:
        title (str): The title value.

    Returns:
        str: The computed or requested result.
    """
    return hashlib.sha1(_normalize_text(title).encode("utf-8")).hexdigest()


def article_id(canonical_url: str) -> str:
    """Article id for news collection, tagging, deduplication, and storage utilities.

    Args:
        canonical_url (str): The canonical url value.

    Returns:
        str: The computed or requested result.
    """
    return hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 3) -> list[str]:
    """Shingles for news collection, tagging, deduplication, and storage utilities.

    Args:
        text (str): The text value.
        k (int): The k value. Defaults to ``3``.

    Returns:
        list[str]: The computed or requested result.
    """
    toks = _normalize_text(text).split()
    if len(toks) < k:
        return toks[:]
    return [" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]


def simhash64(text: str, k: int = 3) -> int:
    """Cheap 64-bit SimHash over word shingles. Good enough for
    catching syndicated near-duplicates; not cryptographic.

    Args:
        text (str): The text value.
        k (int): The k value. Defaults to ``3``.

    Returns:
        int: The computed or requested result.
    """

    vec = [0] * 64
    for sh in _shingles(text, k=k):
        h = int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            vec[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, v in enumerate(vec):
        if v > 0:
            out |= 1 << i
    return out


def hamming64(a: int, b: int) -> int:
    """Hamming64 for news collection, tagging, deduplication, and storage utilities.

    Args:
        a (int): The a value.
        b (int): The b value.

    Returns:
        int: The computed or requested result.
    """
    return (a ^ b).bit_count()


def hashes_for(title: str, body: str | None) -> DedupHashes:
    """Hashes for for news collection, tagging, deduplication, and storage utilities.

    Args:
        title (str): The title value.
        body (str | None): The body value.

    Returns:
        DedupHashes: The computed or requested result.
    """
    blob = title if not body else f"{title}\n{body}"
    return DedupHashes(
        title_hash=title_hash(title),
        shingle_hash=format(simhash64(blob), "016x"),
    )
