"""DOI normalization utilities."""


def normalize_doi(s: str | None) -> str | None:
    """Strip the https://doi.org/ prefix and lowercase. Returns None for empty."""
    return (s or "").replace("https://doi.org/", "").lower() or None
