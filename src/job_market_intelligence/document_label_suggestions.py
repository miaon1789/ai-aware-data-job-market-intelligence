"""Private first-pass label suggestions for document-level annotation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentLabelSuggestion:
    role: str
    entry_fit: str
    ai_signal: str
    coding_signal: str
    confidence: str
    evidence: str
    needs_review: bool

    @property
    def labels(self) -> list[str]:
        return [
            f"ROLE::{self.role}",
            f"ENTRY_FIT::{self.entry_fit}",
            f"AI_SIGNAL::{self.ai_signal}",
            f"CODING_SIGNAL::{self.coding_signal}",
        ]


ENTRY_TITLE_RE = re.compile(
    r"\b(?:graduate|junior|entry[- ]?level|intern|internship|student|trainee|"
    r"associate)\b",
    re.IGNORECASE,
)
ENTRY_TEXT_RE = re.compile(
    r"\b(?:recent graduate|graduates?|early career|0\s*[-–]\s*2 years?|"
    r"1\s*[-–]\s*2 years?|one to two years|no prior experience|training provided)\b",
    re.IGNORECASE,
)
EXPERIENCED_RE = re.compile(
    r"\b(?:senior|lead|manager|principal|staff|architect|director|head|executive|"
    r"5\s*[-–]\s*8 years?|7 years?|5\+ years?|3\+ years?|experienced)\b",
    re.IGNORECASE,
)
EXPERIENCE_YEARS_RE = re.compile(
    r"\b(?:3|4|5|6|7|8|9|10)\+?\s*(?:[-–]\s*\d+)?\s+years?\b",
    re.IGNORECASE,
)

EXPLICIT_AI_RE = re.compile(
    r"\b(?:chatgpt|chat gpt|claude(?: code)?|copilot|cursor|gemini|genai|"
    r"generative ai|large language models?|llms?|rag|retrieval[- ]augmented "
    r"generation|langchain|prompt engineering|prompt design|agentic|ai agents?|"
    r"multi[- ]agent|conversational ai)\b",
    re.IGNORECASE,
)
EXPLICIT_AI_ML_RE = re.compile(
    r"\b(?:machine learning|deep learning|computer vision|natural language "
    r"processing|nlp|ml engineer|machine learning engineer|ai engineer|"
    r"ai engineering|applied ai|ai systems?|ai platform|ai accelerator|"
    r"model training|model evaluation|model serving|mlops|reinforcement "
    r"learning|ai research)\b",
    re.IGNORECASE,
)
WORKFLOW_AUTOMATION_RE = re.compile(
    r"\b(?:workflow automation|process automation|automation workflows?|"
    r"automated workflows?|automating|automation|n8n|zapier|crm automation)\b",
    re.IGNORECASE,
)
AI_WORKFLOW_AUTOMATION_RE = re.compile(
    r"\b(?:agent workflows?|ai-assisted workflows?|workflow automation|"
    r"process automation|n8n|zapier|crm automation)\b",
    re.IGNORECASE,
)
VAGUE_AI_RE = re.compile(
    r"\b(?:ai[- ]powered|ai[- ]enabled|ai[- ]driven|artificial intelligence|"
    r"\bai\b|intelligent applications?|intelligent automation)\b",
    re.IGNORECASE,
)
AI_BOILERPLATE_RE = re.compile(
    r"\b(?:salesforce is the .*ai crm|as a leader in ai|ai[- ]infused "
    r"(?:scenario planning|platform)|powered by agentic ai|we'?re investing "
    r"in data, digital and ai|data, digital & ai|ai factory|ai factories)\b",
    re.IGNORECASE,
)
PHYSICAL_INFRA_RE = re.compile(
    r"\b(?:data centre construction|data center construction|datacentre|"
    r"datacenter|dc design|autocad|mep services|mechanical|electrical|"
    r"construction|civil engineering|quantity take[- ]offs?)\b",
    re.IGNORECASE,
)

DATA_ENGINEER_RE = re.compile(
    r"\b(?:data engineer|analytics engineer|data platform|data pipeline|"
    r"data pipelines|etl|elt|warehouse|lakehouse|databricks|snowflake|dbt|"
    r"airflow|spark|fabric|medallion|data architecture)\b",
    re.IGNORECASE,
)
DATA_SCIENCE_RE = re.compile(
    r"\b(?:data scientist|machine learning|mlops|predictive model|predictive "
    r"modelling|predictive modeling|statistical model|quant analytics|"
    r"forecasting|model evaluation)\b",
    re.IGNORECASE,
)
AI_ROLE_RE = re.compile(
    r"\b(?:ai engineer|ai automation|ai adoption|ai solution|ai solutions|"
    r"ai security|generative ai|gen ai|llm|prompt engineer|conversational ai|"
    r"ai product|ai transformation|ai consultant)\b",
    re.IGNORECASE,
)
ANALYST_RE = re.compile(
    r"\b(?:data analyst|analytics analyst|business intelligence|bi analyst|"
    r"reporting analyst|campaign data analyst|insights analyst|business analyst|"
    r"data analytics|dashboard|dashboards|power bi|tableau|reporting|insights)\b",
    re.IGNORECASE,
)
NON_DATA_ANALYST_RE = re.compile(
    r"\b(?:financial analyst|finance|investor relations|brand analyst|energy "
    r"trading|tax technology|property fund|fp&a|estimator|marketing associate)\b",
    re.IGNORECASE,
)

PRODUCTION_CODING_RE = re.compile(
    r"\b(?:software engineer|full stack|frontend|front-end|backend|back-end|"
    r"devsecops|ci/cd|playwright|api|apis|deployment|production|monitoring|"
    r"reliability|docker|kubernetes|"
    r"terraform|next\.js|react|typescript|\.net|node\.js|django|pipeline|"
    r"pipelines|mlops|data engineer|machine learning engineer|ai engineer)\b",
    re.IGNORECASE,
)
NO_PRODUCTION_CODING_ROLE_RE = re.compile(
    r"\b(?:business analyst|technical business analyst|solution architect|"
    r"solutions architect|pre[- ]sales|sales specialist|account executive|"
    r"network engineer|network engineering|systems engineer|clinical lead|"
    r"manager, ai infrastructure|data enablement|recruitment consultant)\b",
    re.IGNORECASE,
)
PROTOTYPE_CODING_RE = re.compile(
    r"\b(?:prototype|prototyping|proof[- ]of[- ]value|proof[- ]of[- ]concept|"
    r"\bpoc\b|build .*agents?|agent workflows?)\b",
    re.IGNORECASE,
)
AUTOMATION_CODING_RE = re.compile(
    r"\b(?:script(?:ing|ed)?|macros?|vba|n8n|zapier|agent workflows?|"
    r"prototype|prototyping|proof[- ]of[- ]value|proof[- ]of[- ]concept|"
    r"\bpoc\b)\b",
    re.IGNORECASE,
)
STANDALONE_CODING_RE = re.compile(
    r"\b(?:python|sql|r language|programming in r|java|javascript|typescript|"
    r"c#|\.net|scala|bash|pyspark)\b",
    re.IGNORECASE,
)


def title_from_text(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _has(pattern: re.Pattern[str], value: str) -> bool:
    return bool(pattern.search(value))


def suggest_document_labels(text: str) -> DocumentLabelSuggestion:
    title = title_from_text(text)
    body = text.split("\n", 1)[1] if "\n" in text else ""
    combined = f"{title}\n{text}"
    coding_text = body or text
    title_lower = title.lower()

    role = "Other / Mixed"
    role_evidence = "no strong role-family signal"
    if _has(AI_ROLE_RE, combined):
        role = "AI / Automation"
        role_evidence = "AI/automation role signal"
    elif _has(DATA_ENGINEER_RE, combined):
        role = "Data Engineer"
        role_evidence = "data engineering or platform signal"
    elif _has(DATA_SCIENCE_RE, combined):
        role = "Data Scientist / ML"
        role_evidence = "data science, ML or modelling signal"
    elif _has(ANALYST_RE, combined) and not (
        _has(NON_DATA_ANALYST_RE, title) and not re.search(r"\bdata\b", title_lower)
    ):
        role = "Data Analyst / BI"
        role_evidence = "analytics, BI, reporting or business-analysis signal"

    title_entry = _has(ENTRY_TITLE_RE, title)
    text_entry = _has(ENTRY_TEXT_RE, combined)
    experienced = _has(EXPERIENCED_RE, title) or _has(EXPERIENCE_YEARS_RE, combined)
    if experienced and not title_entry:
        entry_fit = "Experienced role"
        entry_evidence = "senior/lead/manager or 3+ years signal"
    elif title_entry or text_entry:
        entry_fit = "Likely entry-level"
        entry_evidence = "graduate/junior/intern/entry-level signal"
    elif experienced:
        entry_fit = "Experienced role"
        entry_evidence = "experience signal"
    else:
        entry_fit = "Unclear"
        entry_evidence = "no reliable entry-level or experienced signal"

    has_physical_infra_noise = _has(PHYSICAL_INFRA_RE, combined)
    has_ai_boilerplate = _has(AI_BOILERPLATE_RE, combined)
    title_has_ai = bool(
        re.search(r"\b(?:ai|ml|machine learning|genai|gen ai)\b", title, re.IGNORECASE)
    )

    if _has(EXPLICIT_AI_RE, combined):
        ai_signal = "Explicit GenAI / LLM tool"
        ai_evidence = "explicit GenAI, LLM, agent or named AI-tool signal"
    elif _has(EXPLICIT_AI_ML_RE, combined) and not has_physical_infra_noise:
        ai_signal = "Explicit AI / ML system work"
        ai_evidence = "explicit AI, ML, model, computer-vision or AI-engineering signal"
    elif (
        _has(AI_WORKFLOW_AUTOMATION_RE, combined)
        and (
            title_has_ai
            or re.search(
                r"\b(?:n8n|zapier|ai-assisted|agent)\b", combined, re.IGNORECASE
            )
        )
    ):
        ai_signal = "Workflow automation"
        ai_evidence = "AI-assisted workflow or process automation signal"
    elif has_ai_boilerplate and not title_has_ai:
        ai_signal = "No AI signal"
        ai_evidence = "AI wording appears only in employer or product boilerplate"
    elif _has(VAGUE_AI_RE, combined):
        ai_signal = "AI-adjacent but vague"
        ai_evidence = "role-specific but generic AI wording"
    else:
        ai_signal = "No AI signal"
        ai_evidence = "no AI or automation signal"

    no_production_coding_role = _has(NO_PRODUCTION_CODING_ROLE_RE, combined)
    if _has(PRODUCTION_CODING_RE, coding_text) and not no_production_coding_role:
        coding_signal = "Production engineering"
        coding_evidence = "production, engineering, cloud, API or pipeline signal"
    elif _has(PROTOTYPE_CODING_RE, coding_text) or _has(
        AUTOMATION_CODING_RE, coding_text
    ):
        coding_signal = "Automation / scripting"
        coding_evidence = "automation, workflow or prototype-building signal"
    elif _has(STANDALONE_CODING_RE, coding_text):
        coding_signal = "Standalone coding skill"
        coding_evidence = "language or SQL signal without stronger engineering framing"
    else:
        coding_signal = "No coding signal"
        coding_evidence = "no concrete coding signal"

    uncertain_dimensions = sum(
        [
            role == "Other / Mixed",
            entry_fit == "Unclear",
            ai_signal in {"AI-adjacent but vague", "Unclear"},
            coding_signal == "No coding signal",
        ]
    )
    confidence = "high" if uncertain_dimensions <= 1 else "medium"
    if uncertain_dimensions >= 3:
        confidence = "low"
    needs_review = confidence != "high" or role == "Other / Mixed"
    evidence = "; ".join(
        [role_evidence, entry_evidence, ai_evidence, coding_evidence]
    )
    return DocumentLabelSuggestion(
        role=role,
        entry_fit=entry_fit,
        ai_signal=ai_signal,
        coding_signal=coding_signal,
        confidence=confidence,
        evidence=evidence,
        needs_review=needs_review,
    )
