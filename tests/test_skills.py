from job_market_intelligence.skills import SkillExtractor


def contexts(text: str) -> dict[str, str]:
    return {item.skill: item.context for item in SkillExtractor().extract(text)}


def test_extracts_and_normalises_aliases():
    result = contexts("Strong experience with PostgreSQL and PowerBI is essential.")
    assert result["SQL"] == "required"
    assert result["Power BI"] == "required"


def test_negation_overrides_required_marker():
    result = contexts("Python is not required for this graduate role.")
    assert result["Python"] == "negated"


def test_preferred_context():
    result = contexts("Experience with AWS and Docker is highly regarded.")
    assert result["AWS"] == "preferred"
    assert result["Docker"] == "preferred"
