"""
Utilities for graph reconstruction (label handling, constants, helpers).
"""

from typing import Optional


def sanitize_label(label: Optional[str]) -> str:
    """
    Normalize labels for consistent matching and downstream rendering.
    - Lowercase
    - Replace spaces and hyphens with underscores
    - Collapse repeated underscores
    - Strip leading/trailing underscores
    """
    if not isinstance(label, str):
        return ""
    s = label.strip().lower().replace(" ", "_").replace("-", "_")
    # Collapse multiple underscores
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def canonicalize_seed_label(seed: str) -> str:
    """
    Return a canonical seed label to improve initial matching with the motif library.
    Keep human-readable spacing here; we sanitize later in pruning.
    """
    if isinstance(seed, str) and "upzoning" in seed.lower():
        return "support for upzoning"
    return seed


def is_stance_like_label(label: str) -> bool:
    """
    Determine whether a label is semantically the upzoning stance/support node.
    """
    if not isinstance(label, str):
        return False
    l = label.lower()
    return ("upzoning" in l) and ("stance" in l or "support" in l)
