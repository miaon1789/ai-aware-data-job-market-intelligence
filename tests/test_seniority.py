from job_market_intelligence.seniority import classify_seniority


def test_graduate_signal():
    label, evidence = classify_seniority("Graduate Data Analyst", "Training is provided.")
    assert label == "Graduate / Junior"
    assert evidence == "graduate"


def test_senior_title_takes_precedence_over_graduate_program_wording():
    label, _ = classify_seniority(
        "Senior Data Scientist", "You will mentor graduate team members."
    )
    assert label == "Non-Junior"
