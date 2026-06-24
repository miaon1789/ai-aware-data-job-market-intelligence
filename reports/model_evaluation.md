# Model and Extraction Evaluation

## Australian Adzuna snapshot

The private collection was accessed on 22 June 2026 through the Adzuna API.
After a 90-day age filter and conservative near-duplicate removal, 311
advertisement excerpts remain.

The API description field is not a full job description:

- 94.86% of retained descriptions are exactly 500 characters;
- median description length is approximately 75 words;
- only 24.12% of retained excerpts contain a dictionary skill;
- 101 of 104 extracted mentions have only `mentioned` context.

Consequently, this snapshot supports excerpt-level mention analysis only. It
does not support reliable conclusions about complete required, preferred,
alternative or negated skill requirements.

Role groups in the current aggregate dashboard are provisional predictions from
a classifier trained on synthetic fixtures. They are not evaluation results and
must be replaced or validated with the frozen private human-annotation sample.

See `data/public/metadata.json` and `reports/data_quality.json` for machine-
readable provenance and limitations.

## SkillSpan public benchmark

Dataset: SkillSpan, pinned commit
`2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b`, MIT License.

Task: sentence-level detection of whether any human-annotated skill or knowledge
span is present. This is deliberately narrower than exact span evaluation.

| Subset | Samples | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 3,569 | 0.8314 | 0.1468 | 0.2496 |
| Tech | 2,286 | 0.8288 | 0.2188 | 0.3462 |
| House | 1,283 | 0.8462 | 0.0523 | 0.0984 |

Interpretation:

- precision is relatively strong, so detected dictionary terms are usually
  associated with an annotated skill/knowledge sentence;
- recall is poor, confirming that the current 36-entry dictionary is too narrow;
- the gap between Tech and House shows that the taxonomy is strongly biased
  toward technical tools;
- accuracy is not a useful headline metric because most benchmark sentences are
  negative.

The next modelling priority is taxonomy expansion with error analysis, followed
by an exact-span baseline. Required/preferred context evaluation remains blocked
until suitable full-text, context-labelled data are available.
