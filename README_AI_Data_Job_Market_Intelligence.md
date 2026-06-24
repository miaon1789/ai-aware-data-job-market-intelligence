# AI & Data Job Market Intelligence  
### Skill Mining, Role Segmentation and AI-Era Hiring Signals in Australian Tech Job Ads

> An end-to-end data and NLP project for analysing advertised skill demand in Australian AI, data and IT roles, with a focus on Sydney vs Melbourne, graduate/junior opportunities, AI-related skills, and role-specific skill gaps.

---

## 1. Project Overview

The rapid adoption of AI-assisted programming tools, large language models and automation platforms is changing how technical work is described in job advertisements. For graduates and career changers, it is increasingly difficult to understand which skills are genuinely required, which are merely preferred, and how different roles such as Data Analyst, Data Scientist, Data Engineer and ML/AI Engineer differ in practice.

This project builds a job-market intelligence system that collects public job-ad data, extracts skills and role signals from unstructured job descriptions, classifies role families and seniority levels, and visualises the results through an interactive dashboard.

The project is designed primarily for **job seekers**, not employers. It does not rank candidates or automate hiring decisions. Instead, it helps users understand advertised labour-market demand and identify skill combinations worth developing.

---

## 2. Research Questions

This project investigates the following questions:

1. How do AI, data and IT job advertisements differ between Sydney and Melbourne?
2. How do graduate/junior roles differ from mid-level and senior roles in terms of skills, experience and education requirements?
3. How frequently do skills such as Python, SQL, cloud platforms, BI tools, LLMs, MLOps and Docker appear in job advertisements?
4. Which skills commonly appear together, such as Python + SQL + AWS?
5. How do Data Analyst, Data Scientist, Data Engineer and ML/AI Engineer roles differ in their advertised skill structures?
6. How are AI-related skills described in job ads: as core requirements, desirable skills, automation tools or generic buzzwords?
7. For a target role, which skill combinations appear most useful for a graduate job seeker to prioritise?

---

## 3. Key Features

### Market Overview

- Compare job counts across cities, role families and seniority levels.
- Track advertised demand over time where collection dates are available.
- Distinguish between graduate/junior roles and non-junior roles.
- Show role distribution across Sydney and Melbourne.

### Skill Explorer

- Extract and normalise technical skills from job descriptions.
- Compare skill frequencies by city, role family and seniority level.
- Identify common skill combinations and co-occurrence patterns.
- Distinguish between required, preferred, mentioned and negated skills where possible.

### Role Segmentation

- Use text embeddings and clustering to identify common job-ad themes.
- Classify job descriptions into role families such as:
  - Data Analyst
  - Data Scientist
  - Data Engineer
  - ML/AI Engineer
  - Software Engineer
  - Other / Mixed
- Analyse how role boundaries overlap in real job advertisements.

### JD Analyser

Users can paste a job description into the dashboard and receive:

- Predicted role family
- Predicted seniority level
- Extracted required and preferred skills
- AI-related skill mentions
- Similarity to market benchmarks
- Common skills missing from the pasted job description compared with similar roles

Example output:

```text
This job description is most similar to Junior Data Scientist roles in Sydney.
Estimated similarity: 78%

Extracted required skills:
Python, SQL, machine learning, communication

Extracted preferred skills:
AWS, Docker, Power BI

Common skills in similar roles but not found in this description:
Git, data visualisation, statistics, stakeholder communication
```

---

## 4. End-to-End Architecture

```text
Public job data / API
        ↓
Scheduled or manual ingestion
        ↓
Raw data validation and deduplication
        ↓
Text cleaning and normalisation
        ↓
Role and seniority extraction
        ↓
NLP skill/entity extraction
        ↓
Role classification and clustering
        ↓
SQL analytics layer
        ↓
Streamlit dashboard and JD analyser
        ↓
Evaluation, monitoring and reporting
```

---

## 5. Data Sources

The project is designed to work with public or permitted job-ad data sources.

Potential sources include:

- Public job-search APIs, subject to API terms and attribution requirements.
- Open research datasets or Kaggle datasets containing job advertisements.
- Public company career pages, only where access is permitted and collection respects robots.txt, rate limits and website terms.

This project does **not** scrape restricted job platforms unless explicit permission or a suitable API/license is available.

### Data Access and Publication Policy

To reduce data-use risk, this repository does not publish large collections of raw job descriptions unless the data license clearly permits redistribution.

The public repository is intended to contain:

- Data schema
- Data ingestion code
- Synthetic or small sample data
- Aggregated statistics
- Dashboard code
- Model training and evaluation code
- Documentation for users to provide their own API keys or datasets

Raw job-ad text, if collected under API or website terms that restrict redistribution, should be stored locally or in private storage and excluded from GitHub.

---

## 6. Tech Stack

### Data Processing

- Python
- pandas / Polars
- Pydantic
- DuckDB or PostgreSQL

### NLP and Machine Learning

- spaCy
- scikit-learn
- Sentence Transformers
- TF-IDF
- Logistic Regression / Linear SVM / Random Forest
- KMeans / HDBSCAN / BERTopic, optional

### Dashboard and App

- Streamlit
- Plotly
- FastAPI, optional

### Engineering

- Docker
- GitHub Actions
- pytest
- Ruff
- Makefile or task runner

---

## 7. NLP Design

The NLP pipeline uses a three-layer design.

### 7.1 Skill Dictionary and Normalisation

A curated skill dictionary maps variants into standard skill names.

Examples:

```text
PostgreSQL, MySQL, SQL Server → SQL
Amazon Web Services, AWS → AWS
PowerBI, Power BI → Power BI
Large Language Model, LLM, GenAI → Generative AI
```

Each skill is assigned to a broader category, such as:

- Programming
- Data analysis
- Machine learning
- Cloud
- Data engineering
- BI and visualisation
- MLOps
- AI and LLM tools
- Soft skills
- Domain knowledge

### 7.2 Context Rules

The project does not rely only on keyword counts. It also attempts to classify how a skill is mentioned.

Examples:

```text
"Python required" → required
"Python desirable" → preferred
"Python is not required" → negated
"experience with Python or R" → alternative / mentioned
```

Skill mentions are classified into:

- `required`
- `preferred`
- `mentioned`
- `negated`
- `alternative`

### 7.3 Embedding and Model Layer

Sentence embeddings are used for:

- Role clustering
- Job-description similarity
- JD analyser benchmarking
- Role-family classification features

A supervised classifier is trained on manually labelled job descriptions to predict:

- Role family
- Seniority level

---

## 8. Machine Learning Tasks

### Task 1: Role Family Classification

**Input:** job title and job description  
**Output:** predicted role family

Target classes include:

```text
Data Analyst
Data Scientist
Data Engineer
ML/AI Engineer
Software Engineer
Other / Mixed
```

Evaluation metrics:

- Accuracy
- Macro-F1
- Per-class precision and recall
- Confusion matrix

### Task 2: Seniority Classification

**Input:** job title and job description  
**Output:** seniority level

Target classes:

```text
Graduate / Junior
Mid-level
Senior
Lead / Manager
Unclear
```

Evaluation metrics:

- Precision
- Recall
- F1-score
- Error analysis for ambiguous roles

### Task 3: Skill Extraction Evaluation

A manually labelled evaluation set is used to test skill extraction quality.

Evaluation metrics:

- Skill-level precision
- Skill-level recall
- Skill-level F1
- Context classification accuracy for required/preferred/negated mentions

---

## 9. Dashboard Pages

The Streamlit dashboard contains four main pages.

### 9.1 Market Overview

Shows:

- Job count by city
- Job count by role family
- Junior vs non-junior role distribution
- Posting trends over time
- Sydney vs Melbourne comparison

### 9.2 Skill Explorer

Shows:

- Most frequent skills
- Skill frequency by role
- Skill frequency by city
- Required vs preferred skills
- Skill co-occurrence network
- Heatmap of role family × skill category

### 9.3 Role Segments

Shows:

- Job-ad clusters based on text embeddings
- Cluster keywords and representative job titles
- Role overlap between data, software and AI positions
- AI-related role segments

### 9.4 JD Analyser

Allows users to paste a job description and returns:

- Predicted role family
- Predicted seniority
- Extracted skills
- Required/preferred/negated skill classification
- Similarity to market benchmark roles
- Missing common skills compared with similar roles

---

## 10. Repository Structure

```text
ai-aware-data-job-market-intelligence/
│
├── README.md
├── pyproject.toml
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/                  # excluded from GitHub if license-restricted
│   ├── processed/            # cleaned data or local outputs
│   ├── public/               # aggregate-only public release
│   └── external/             # external reference files
│
├── src/
│   └── job_market_intelligence/
│
├── app/
│   ├── streamlit_app.py
│   └── public_dashboard.py
│
├── tests/
│   └── test_*.py
│
├── reports/
│   ├── *.json
│   ├── *.csv
│   └── model_evaluation.md
│
└── scripts/
    ├── run_pipeline.py
    └── *.py
```

---

## 11. Installation

Clone the repository:

```bash
git clone https://github.com/miaon1789/ai-aware-data-job-market-intelligence.git
cd ai-aware-data-job-market-intelligence
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

---

## 12. How to Run

### Run the full pipeline

```bash
python scripts/run_pipeline.py
```

### Run tests

```bash
pytest
```

### Run code quality checks

```bash
ruff check .
```

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 13. Current Scope

The first version focuses on:

- Sydney and Melbourne
- Recent job advertisements, depending on available data
- Four main role families:
  - Data Analyst
  - Data Scientist
  - Data Engineer
  - ML/AI Engineer
- Graduate/junior vs non-junior segmentation
- 50–100 standardised technical and professional skills
- One Streamlit dashboard
- One manually labelled evaluation set for NLP and classification testing

---

## 14. Evaluation Plan

The project evaluates both analytical outputs and model performance.

### Skill Extraction

```text
Precision = correctly extracted skills / all extracted skills
Recall = correctly extracted skills / all labelled skills
F1 = harmonic mean of precision and recall
```

### Role Classification

Metrics:

- Accuracy
- Macro-F1
- Per-class precision and recall
- Confusion matrix

### Seniority Classification

Metrics:

- Binary F1 for Graduate/Junior vs Non-Junior
- Multi-class F1 for detailed seniority labels
- Error analysis for ambiguous job ads

### Dashboard Validation

The dashboard is evaluated qualitatively by checking whether it can answer practical job-market questions, such as:

- What skills are most common in junior data roles?
- Which skills differentiate Data Engineer from Data Analyst roles?
- How common are LLM-related skills in AI/ML job ads?
- Does Sydney advertise more ML/AI roles than Melbourne in the collected sample?
- Which missing skills are most relevant for a pasted job description?

---

## 15. Limitations

This project analyses **advertised demand**, not the full labour market. Job advertisements may not perfectly reflect actual hiring criteria, internal promotion pipelines, recruiter screening behaviour or final hiring decisions.

Key limitations include:

- Job ads may be duplicated across platforms.
- Some job descriptions are vague or use inflated skill lists.
- “Junior” roles are not always labelled clearly in job titles.
- Skill mentions do not always indicate actual importance.
- API or dataset coverage may be incomplete.
- Short collection windows cannot support strong long-term trend claims.
- AI-related skill demand should be interpreted as advertised language, not direct evidence of causal labour-market change.

The project therefore avoids claims such as “AI caused a decline in programming jobs” unless supported by suitable longitudinal data and causal analysis.

---

## 16. Ethical and Legal Considerations

This project is designed for labour-market understanding and job-seeker education.

It should not be used for:

- Candidate ranking
- Automated hiring decisions
- Discriminatory screening
- Candidate document screening
- Individual employment predictions

Data use principles:

- Respect API terms, website terms and robots.txt.
- Do not redistribute raw job descriptions unless permitted.
- Store API keys securely and never commit them to GitHub.
- Publish aggregate statistics rather than restricted raw text where required.
- Clearly distinguish between descriptive analysis and causal claims.
- Clearly label results as based on collected job advertisements, not the entire job market.

---

## 17. Roadmap

### Version 1: Minimum Viable Public Release

- [ ] Build data schema
- [ ] Collect or load permitted job-ad data
- [ ] Clean and deduplicate job ads
- [ ] Build skill dictionary
- [ ] Implement rule-based skill extraction
- [ ] Implement role-family classification
- [ ] Implement seniority classification
- [ ] Build Streamlit dashboard
- [ ] Add JD analyser
- [ ] Add tests and documentation
- [ ] Add model evaluation report

### Version 2: Stronger NLP and Evaluation

- [ ] Add manually labelled evaluation set of 300–500 job descriptions
- [ ] Add Sentence Transformer embeddings
- [ ] Add role clustering
- [ ] Add HDBSCAN or BERTopic topic discovery
- [ ] Compare rule-based extraction with ML/LLM-assisted extraction
- [ ] Add skill co-occurrence network

### Version 3: Monitoring and Trend Analysis

- [ ] Add scheduled ingestion
- [ ] Add data freshness checks
- [ ] Add longitudinal trend charts
- [ ] Add AI-skill demand index
- [ ] Add monthly reporting
- [ ] Add deployment pipeline

---

## 18. License

See [`LICENSE`](LICENSE).

Note: The code may be open sourced, but raw job-ad data may be excluded from the repository depending on data-source terms and licensing restrictions.

---

## 19. Acknowledgements

This project is inspired by the practical challenges faced by graduate job seekers entering AI, data and software roles in a rapidly changing labour market.

Vacancy data are sourced from The Adzuna API. Public outputs are aggregate-only
and retain Adzuna attribution. The optional SkillSpan benchmark is pinned to an
openly licensed upstream dataset for reproducible skill-extraction evaluation.
