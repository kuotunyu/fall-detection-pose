"""Pure helpers shared by demo media tooling."""

from __future__ import annotations


def parse_viewport(value: str) -> dict[str, int]:
    """Parse ``WIDTHxHEIGHT`` into a Playwright viewport mapping."""

    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("viewport must use WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("viewport dimensions must be positive")
    return {"width": width, "height": height}
