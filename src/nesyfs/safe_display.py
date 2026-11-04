"""Create a display-safe copy of stored and live data without rewriting it."""
from __future__ import annotations

import html
import json
import re
import unicodedata

WITHHELD = "[withheld]"
_TAG = re.compile(
    r"<\s*\|?\s*/?\s*(?:t\s*h\s*i\s*n\s*k|a\s*n\s*a\s*l\s*y\s*s\s*i\s*s|"
    r"r\s*e\s*a\s*s\s*o\s*n\s*i\s*n\s*g)", re.IGNORECASE
)
_UNICODE_ESCAPE = re.compile(r"\\+u([0-9a-f]{4})", re.IGNORECASE)
_HEX_ESCAPE = re.compile(r"\\+x([0-9a-f]{2})", re.IGNORECASE)
_PRIVATE_JSON_KEY = re.compile(
    r'''["']\s*(?:raw[\w-]*|[\w-]*(?:reasoning|thinking|analysis|chain[_ -]?of[_ -]?thought)[\w-]*)\s*["']\s*:''',
    re.IGNORECASE,
)


def _private_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return (normalized.startswith("raw") or "reasoning" in normalized
            or "thinking" in normalized or "chainofthought" in normalized
            or "analysis" in normalized)


def _marked(text: str) -> bool:
    # Legacy JSON may escape tags repeatedly, or contain HTML character entities.
    candidate = text
    for _ in range(4):
        candidate = unicodedata.normalize("NFKC", html.unescape(candidate))
        candidate = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", candidate)
        candidate = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), candidate)
        candidate = _HEX_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), candidate)
        if _TAG.search(candidate):
            return True
    return False


def safe_display(value, _depth: int = 0):
    """Return a sanitized copy for JSON responses; never mutate the stored data."""
    if _depth > 40:
        return WITHHELD
    if isinstance(value, dict):
        return {key: safe_display(item, _depth + 1) for key, item in value.items()
                if not (isinstance(key, str) and _private_key(key))}
    if isinstance(value, (list, tuple)):
        return [safe_display(item, _depth + 1) for item in value]
    if isinstance(value, str):
        if _marked(value):
            return WITHHELD
        # A historical event can contain JSON encoded inside a string, including
        # multiple JSON encoding layers. Filter keys there too, preserving its type.
        if value.strip().startswith(('{', '[', '"')):
            try:
                parsed = json.loads(value)
            except (ValueError, RecursionError):
                if _PRIVATE_JSON_KEY.search(value):
                    return WITHHELD
            else:
                if isinstance(parsed, (dict, list)) or (isinstance(parsed, str) and parsed != value):
                    return json.dumps(safe_display(parsed, _depth + 1), ensure_ascii=False)
        return value
    return value
