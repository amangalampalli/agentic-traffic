"""Validation helpers for URL path parameters used to construct filesystem paths."""
from __future__ import annotations

import re

from fastapi import HTTPException

# Allow letters, digits, underscores, hyphens, and dots (for e.g. "city_01.v2").
# Disallow anything that could traverse directories: slashes, null bytes, etc.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_\-\.]+$")
_MAX_SEGMENT_LEN = 128


def validate_path_segment(value: str, field: str) -> str:
    """Raise HTTP 400 if *value* is not a safe filesystem path component."""
    if not value:
        raise HTTPException(status_code=400, detail=f"{field} must not be empty.")
    if len(value) > _MAX_SEGMENT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"{field} exceeds maximum length of {_MAX_SEGMENT_LEN} characters.",
        )
    if not _SAFE_SEGMENT.match(value):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field} contains invalid characters. "
                "Only letters, digits, underscores, hyphens, and dots are allowed."
            ),
        )
    return value
