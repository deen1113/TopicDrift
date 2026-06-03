"""Cache-key utilities."""

import hashlib


def make_cache_key(key_material: str | list[str]) -> str:
    """SHA-1 hex digest of a string or sorted join of a string list."""
    if isinstance(key_material, list):
        s = "|".join(sorted(key_material))
    else:
        s = key_material
    return hashlib.sha1(s.encode()).hexdigest()
