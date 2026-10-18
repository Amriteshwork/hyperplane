# Content

Learning content is code: Markdown + YAML, reviewed in PRs, validated by
`uv run python -m scripts.content_check` (from `backend/`), and indexed by the API at startup.

```
content/
  algorithms/<algo_id>/        one folder per algorithm pack (id = pack id, e.g. linear_regression)
    overview.md                frontmatter + required sections (below)
    resources.yaml             curated resources specific to this algorithm
  concepts/<concept_id>/       one folder per concept (kebab-case, e.g. train-test-split)
    concept.md                 short explanation; frontmatter: title, prerequisites[]
    resources.yaml
    posts/*.md                 our own posts, when good external material doesn't exist
  families/<family_id>.md      one page per family (e.g. linear-models.md)
```

## `overview.md`

```markdown
---
title: Linear Regression
summary: Fit a straight line (or hyperplane) that minimises squared error.
difficulty: 1          # 1–5
---

## Intuition
## How it works
## Math
## Assumptions
## Hyperparameters
## When to use / not use
## Common pitfalls
## Interview questions
```

Math uses KaTeX: `$inline$` and `$$display$$`.

## `resources.yaml`

```yaml
- type: video                 # blog | video | paper | interactive | course
  title: "Linear regression, clearly explained"
  url: https://example.com/...
  author: Jane Doe
  level: beginner             # beginner | intermediate | advanced
  duration_min: 12
  why: "One line on why this is worth the learner's time."
  source: curated             # curated | own
  checked: 2026-10-17         # last link check
```

## Rules

- Every concept needs at least one beginner blog/post and one video; every algorithm needs a
  complete overview. Gaps go to the generated content backlog.
- Curation rubric: correct, clear, recent enough for current library APIs, no paywall, matches the
  stated level. One excellent resource beats five average ones.
- Own videos live on YouTube and are referenced like any other video.
