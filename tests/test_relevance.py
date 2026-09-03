import pandas as pd

from job_market_intelligence.relevance import filter_relevant_jobs, is_relevant_title


def test_is_relevant_title_rules():
    # Clearly technical titles are always kept.
    assert is_relevant_title("Graduate Software Engineer", 0)
    assert is_relevant_title("Junior Data Scientist", 0)
    assert is_relevant_title("Senior Backend Engineer", 0)
    # Clearly non-technical professions are dropped even with a stray skill.
    assert not is_relevant_title("Graduate Mental Health Nurse", 1)
    assert not is_relevant_title("New Graduate Radiographer", 0)
    assert not is_relevant_title("Graduate Environmental Planner", 0)
    assert not is_relevant_title("Account Executive - Federal", 0)
    # Ambiguous titles depend on technical skill evidence.
    assert is_relevant_title("Junior Operations Analyst", 3)
    assert not is_relevant_title("Junior Operations Analyst", 1)


def test_filter_relevant_jobs_splits_and_prunes_mentions():
    jobs = pd.DataFrame(
        {
            "job_id": ["1", "2", "3"],
            "title": [
                "Graduate Data Scientist",
                "Graduate Mental Health Nurse",
                "Junior Operations Analyst",
            ],
        }
    )
    skill_mentions = pd.DataFrame(
        {
            "job_id": ["1", "2", "3", "3"],
            "skill": ["Python", "Excel", "SQL", "Python"],
            "category": ["Programming", "Professional", "Programming", "Programming"],
        }
    )
    kept, mentions, dropped = filter_relevant_jobs(jobs, skill_mentions, min_tech_skills=2)
    assert set(kept["job_id"]) == {"1", "3"}  # nurse dropped; analyst kept on 2 tech skills
    assert set(dropped["job_id"]) == {"2"}
    assert set(mentions["job_id"]) == {"1", "3"}  # nurse's mention pruned
