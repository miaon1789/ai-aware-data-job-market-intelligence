from job_market_intelligence.document_label_suggestions import suggest_document_labels


def test_suggests_entry_level_genai_intern_labels():
    suggestion = suggest_document_labels(
        "Generative AI Intern\n\nWork with LLMs, RAG and agent-based AI systems."
    )

    assert suggestion.role == "AI / Automation"
    assert suggestion.entry_fit == "Likely entry-level"
    assert suggestion.ai_signal == "Explicit GenAI / LLM tool"
    assert suggestion.coding_signal == "No coding signal"


def test_suggests_experienced_data_engineering_labels():
    suggestion = suggest_document_labels(
        "Senior Data Engineer\n\nBuild production data pipelines on Azure."
    )

    assert suggestion.role == "Data Engineer"
    assert suggestion.entry_fit == "Experienced role"
    assert suggestion.coding_signal == "Production engineering"


def test_suggests_non_data_analyst_as_mixed():
    suggestion = suggest_document_labels(
        "Junior Financial Analyst\n\nSupport budgeting and finance reporting."
    )

    assert suggestion.role == "Other / Mixed"
    assert suggestion.entry_fit == "Likely entry-level"


def test_does_not_infer_coding_from_data_engineer_title_only():
    suggestion = suggest_document_labels(
        "Data Engineer\n\nAbout the company. We are building a sustainability platform."
    )

    assert suggestion.role == "Data Engineer"
    assert suggestion.coding_signal == "No coding signal"


def test_treats_explicit_ml_as_ai_system_work():
    suggestion = suggest_document_labels(
        "Data Scientist\n\nDevelop Machine Learning and Deep Learning capabilities."
    )

    assert suggestion.ai_signal == "Explicit AI / ML system work"


def test_business_analyst_automation_does_not_imply_coding():
    suggestion = suggest_document_labels(
        "Business Analyst | Manager | Tax Technology\n\n"
        "Help scale a tax automation system across the practice."
    )

    assert suggestion.role == "Other / Mixed"
    assert suggestion.coding_signal == "No coding signal"
