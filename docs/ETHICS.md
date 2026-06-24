# Ethics and Responsible Use

This system is for descriptive labour-market exploration and education.

It must not be used for candidate ranking, candidate document screening, automated hiring,
individual employment predictions, or inferring protected characteristics.

## Product safeguards

- The application processes pasted job descriptions in memory and does not log
  or persist them.
- Results refer to advertisements in the loaded sample, not to a candidate's
  abilities or the whole labour market.
- Skill comparisons include denominators and are descriptive rather than
  prescriptive.
- AI-signal labels describe wording in job-advertisement excerpts. They must not
  be used to claim that an employer actually uses a named AI tool or that a
  candidate should be filtered for tool-specific familiarity.
- Raw text and personal contact details are excluded from public outputs.
- Dataset source, collection period, licence and limitations must be displayed
  alongside results from real data.
- The local analyst dashboard must not be deployed with a real row-level
  database. Public deployment uses the aggregate-only dashboard.
- Real aggregate cells containing fewer than ten advertisements are suppressed.

## Bias and uncertainty

Job advertisements reflect employer language and may reproduce occupational,
gender, disability, age or migration-status biases. A frequent requirement is
not automatically fair, necessary, or useful. Role, entry-fit, AI-signal and
coding-signal labels are uncertain labels for analysis, not facts about a
person, employer or job.
