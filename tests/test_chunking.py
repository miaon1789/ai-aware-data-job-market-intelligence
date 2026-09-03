import pandas as pd

from job_market_intelligence.retrieval.chunking import (
    build_chunk_frame,
    chunk_advertisement,
    classify_heading,
    detect_sections,
    estimate_tokens,
    split_units,
)

STRUCTURED_AD = """About Northwind

Northwind builds payments infrastructure for Australian merchants.

About the role

You will join the data platform team in Sydney.

What you'll do

- Build dbt models and Airflow DAGs
- Partner with analysts on reporting

Requirements

- 3+ years with SQL and Python
- Experience with Databricks or Snowflake

Nice to have

- Exposure to dbt Cloud

Why join us

- Flexible hybrid work

Equal opportunity

Northwind is an equal opportunity employer.
"""


def test_classifies_headings_by_intent_not_keyword_order():
    # "About X" splits three ways depending on what follows it.
    assert classify_heading("About Northwind") == "about"
    assert classify_heading("About the role") == "role"
    assert classify_heading("About you") == "requirements"
    assert classify_heading("Nice to have") == "preferred"
    assert classify_heading("Pay Transparency") == "legal"


def test_body_prose_containing_a_keyword_is_not_a_heading():
    # A requirement bullet mentioning "minimum" must not open a new section.
    assert classify_heading("minimum of 15 hours a week") is None
    assert classify_heading("- Experience with Databricks") is None
    assert classify_heading("You will be expected to deliver reporting on time.") is None


def test_detects_sections_in_structured_text():
    sections = detect_sections(STRUCTURED_AD)
    names = [section.name for section in sections]
    assert names[0] == "about"
    assert "responsibilities" in names
    assert "requirements" in names
    assert "preferred" in names
    assert "legal" in names
    requirements = next(s for s in sections if s.name == "requirements")
    assert "Databricks" in requirements.body


def test_unstructured_excerpt_becomes_one_intro_section():
    excerpt = (
        "Graduate Data Analyst working across SQL and Power BI to support "
        "reporting for a Sydney retailer. Apply now."
    )
    sections = detect_sections(excerpt)
    assert [section.name for section in sections] == ["intro"]
    assert sections[0].body == excerpt


def test_split_units_keeps_bullets_whole_and_splits_sentences():
    units = split_units("- Build dbt models and ship them\nFirst sentence. Second sentence.")
    assert units == [
        "- Build dbt models and ship them",
        "First sentence.",
        "Second sentence.",
    ]


def test_chunks_never_cut_a_bullet_in_half():
    chunks = chunk_advertisement(
        STRUCTURED_AD, job_id="ad-1", title="Data Engineer", target_tokens=64
    )
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.body.split("\n"):
            if line.startswith("- "):
                # Every bullet in the source ends without a trailing hyphen split.
                assert line in STRUCTURED_AD


def test_chunk_text_carries_title_and_heading_for_context():
    chunks = chunk_advertisement(
        STRUCTURED_AD, job_id="ad-1", title="Data Engineer", target_tokens=128
    )
    requirement_chunk = next(c for c in chunks if c.section == "requirements")
    assert requirement_chunk.text.startswith("Data Engineer")
    assert "Requirements" in requirement_chunk.text
    # The citation body stays free of the injected context.
    assert not requirement_chunk.body.startswith("Data Engineer")


def test_chunk_ids_are_stable_and_ordered():
    chunks = chunk_advertisement(STRUCTURED_AD, job_id="ad-1", target_tokens=128)
    assert [c.chunk_id for c in chunks] == [f"ad-1::{i:02d}" for i in range(len(chunks))]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_section_boundaries_bind_before_the_token_budget():
    # Every section in the fixture is shorter than either window, so both
    # settings split at the same places. This is the intended behaviour and it
    # is why the chunk-size ablation only moves on advertisements with sections
    # longer than the window.
    coarse = chunk_advertisement(STRUCTURED_AD, job_id="ad-1", target_tokens=1024)
    fine = chunk_advertisement(STRUCTURED_AD, job_id="ad-1", target_tokens=256)
    assert len(fine) == len(coarse)


def test_window_size_binds_once_a_section_exceeds_it():
    long_section = "Requirements\n" + "\n".join(
        f"- Requirement number {index} covering a named tool and its context"
        for index in range(40)
    )
    coarse = chunk_advertisement(long_section, job_id="ad-1", target_tokens=1024)
    fine = chunk_advertisement(long_section, job_id="ad-1", target_tokens=64)
    assert len(fine) > len(coarse)
    assert all(chunk.section == "requirements" for chunk in fine)


def test_overlap_repeats_trailing_units_between_windows():
    long_section = "Requirements\n" + "\n".join(
        f"- Requirement number {index} covering a named tool" for index in range(30)
    )
    with_overlap = chunk_advertisement(
        long_section, job_id="ad-1", target_tokens=64, overlap_ratio=0.25
    )
    without_overlap = chunk_advertisement(
        long_section, job_id="ad-1", target_tokens=64, overlap_ratio=0.0
    )
    assert len(with_overlap) >= len(without_overlap)
    # Consecutive windows share at least one whole bullet.
    first_lines = set(with_overlap[0].body.split("\n"))
    second_lines = set(with_overlap[1].body.split("\n"))
    assert first_lines & second_lines


def test_oversized_single_unit_is_split_rather_than_dropped():
    long_sentence = " ".join(["word"] * 400)
    chunks = chunk_advertisement(long_sentence, job_id="ad-1", target_tokens=64)
    assert len(chunks) > 1
    assert sum(chunk.body.count("word") for chunk in chunks) >= 400


def test_rejects_invalid_window_settings():
    for kwargs in ({"target_tokens": 8}, {"overlap_ratio": 1.0}, {"overlap_ratio": -0.1}):
        try:
            chunk_advertisement("text", job_id="ad-1", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_build_chunk_frame_flags_truncated_sources():
    jobs = pd.DataFrame(
        [
            {
                "job_id": "full",
                "title": "Data Engineer",
                "description": "flattened copy",
                "description_structured": STRUCTURED_AD,
                "city": "Sydney",
                "company": "Northwind",
                "posted_at": "2026-05-01",
                "source": "greenhouse-ats",
                "analysis_role": "Data Engineer",
                "seniority": "Non-Junior",
            },
            {
                "job_id": "excerpt",
                "title": "Data Analyst",
                "description": "Truncated Adzuna excerpt about SQL reporting work.",
                "description_structured": "Truncated Adzuna excerpt about SQL reporting work.",
                "city": "Melbourne",
                "company": "Harbour Metrics",
                "posted_at": "2026-05-02",
                "source": "adzuna-api",
                "analysis_role": "Data Analyst",
                "seniority": "Unclear",
            },
        ]
    )

    frame = build_chunk_frame(jobs, target_tokens=128)

    assert frame.loc[frame.job_id.eq("excerpt"), "is_truncated"].all()
    assert not frame.loc[frame.job_id.eq("full"), "is_truncated"].any()
    assert (frame.job_id.eq("excerpt")).sum() == 1
    assert (frame.job_id.eq("full")).sum() > 1
    assert set(frame.columns) >= {"chunk_id", "section", "body", "text", "city", "analysis_role"}


def test_build_chunk_frame_requires_core_columns():
    try:
        build_chunk_frame(pd.DataFrame([{"job_id": "a"}]))
    except ValueError as exc:
        assert "title" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_estimate_tokens_is_monotonic():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) < estimate_tokens("a" * 400)
