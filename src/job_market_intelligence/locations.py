"""Australian location normalisation shared by ingestion and collectors.

Real Australian job locations always carry an AU signal (``... Australia``,
``AU - ...`` or a state token), so we require one before accepting a city. This
keeps same-named foreign cities out — Melbourne (Florida), Perth (Scotland),
Sydney (Nova Scotia) — and excludes New Zealand explicitly.
"""

from __future__ import annotations

import re

# Australian cities we normalise to. Multi-word entries are matched as substrings.
AU_CITIES = (
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Canberra",
    "Gold Coast",
    "Newcastle",
    "Wollongong",
    "Hobart",
    "Darwin",
)
REMOTE_AU = "Remote (Australia)"
OTHER_AU = "Other (Australia)"
SUPPORTED_CITIES = frozenset(AU_CITIES) | {REMOTE_AU, OTHER_AU}

# Whole-word tokens that signal an Australian location.
_AU_SIGNAL_TOKENS = {
    "australia",
    "australian",
    "au",
    "nsw",
    "vic",
    "qld",
    "wa",
    "sa",
    "act",
    "tas",
    "nt",
}
# Whole-word tokens that positively identify a non-AU location to exclude.
_NON_AU_TOKENS = {"auckland", "wellington", "christchurch", "nz"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def normalise_au_city(raw: object, *, require_signal: bool = False) -> str | None:
    """Map a raw location string to a canonical AU city, or None if not AU.

    Returns one of ``AU_CITIES``, ``REMOTE_AU`` or ``OTHER_AU``; otherwise None.

    ``require_signal=True`` (used when filtering raw job-board locations) demands
    an explicit AU signal (``Australia``/``AU``/state token) so that same-named
    foreign cities like Melbourne (Florida) or Perth (Scotland) are excluded.
    The default lenient mode accepts an already-canonical AU city name on its
    own, so validating a normalised value is idempotent.
    """

    text = " ".join(str(raw or "").split())
    if not text:
        return None
    low = text.lower()
    tokens = _tokens(low)
    if "new zealand" in low or tokens & _NON_AU_TOKENS:
        return None
    has_signal = bool(tokens & _AU_SIGNAL_TOKENS)
    if require_signal and not has_signal:
        return None
    if "australian capital territory" in low:
        return "Canberra"
    for city in AU_CITIES:
        if city.lower() in low:
            return city
    if has_signal:
        return REMOTE_AU if "remote" in low else OTHER_AU
    return None
