from job_market_intelligence.locations import (
    OTHER_AU,
    REMOTE_AU,
    normalise_au_city,
)


def test_strict_mode_accepts_signalled_au_locations():
    assert normalise_au_city("Sydney, Australia", require_signal=True) == "Sydney"
    assert normalise_au_city("AU - Melbourne", require_signal=True) == "Melbourne"
    assert normalise_au_city("Brisbane QLD", require_signal=True) == "Brisbane"
    assert (
        normalise_au_city("Australian Capital Territory, Australia", require_signal=True)
        == "Canberra"
    )
    assert normalise_au_city("Remote - Australia", require_signal=True) == REMOTE_AU
    assert normalise_au_city("Perth, Australia", require_signal=True) == "Perth"


def test_strict_mode_excludes_foreign_and_ambiguous():
    # Same-named foreign cities carry no AU signal, so they are excluded.
    assert normalise_au_city("Melbourne, Florida", require_signal=True) is None
    assert normalise_au_city("Perth, Scotland", require_signal=True) is None
    assert normalise_au_city("Remote - US", require_signal=True) is None
    assert normalise_au_city("United States", require_signal=True) is None
    assert normalise_au_city("Auckland, New Zealand", require_signal=True) is None
    assert normalise_au_city("", require_signal=True) is None
    assert normalise_au_city("Other City, Australia", require_signal=True) == OTHER_AU


def test_lenient_mode_is_idempotent_on_canonical_labels():
    # Ingestion validates already-normalised values, so bare city names pass.
    assert normalise_au_city("Sydney") == "Sydney"
    assert normalise_au_city("brisbane") == "Brisbane"
    assert normalise_au_city(REMOTE_AU) == REMOTE_AU
    assert normalise_au_city(OTHER_AU) == OTHER_AU
    # Non-AU places are still rejected.
    assert normalise_au_city("London") is None
    assert normalise_au_city("Auckland") is None
