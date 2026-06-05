"""DOI normalization utilities."""


def normalize_doi(s: str | None) -> str | None:
    """Strip the https://doi.org/ prefix and lowercase. Returns None for empty."""
    if not isinstance(s, str):
        return None
    return s.replace("https://doi.org/", "").lower() or None
