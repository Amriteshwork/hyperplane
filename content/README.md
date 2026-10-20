# Content

Learning content is code: Markdown + YAML, reviewed in PRs, validated by
`uv run python -m scripts.content_check` (from `backend/`), and indexed by the API at startup.

```
content/
  algorithms/<algo_id>/        one folder per algorithm pack (id = pack id, e.g. linear_regression)
    overview.md                frontmatter + required sections (below)
    resources.yaml             ids of its curated resources, in display order
  concepts/<concept_id>/       one folder per concept (kebab-case, e.g. train-test-split)
    concept.md                 short explanation; frontmatter: title, prerequisites[]
    resources.yaml
    posts/*.md                 our own posts, when good external material doesn't exist
    code/numpy.py, pytorch.py  the concept from scratch on random data (see below)
    code/annotations.yaml      explanations for their `# @a:<anchor>` comments
  families/<family_id>.md      one page per family (e.g. linear-models.md)
  resources/<resource_id>.yaml one file per curated link, shared by any page (ADR 0014)
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

## Resources

Each link lives once in `resources/<resource_id>.yaml` (kebab-case id = filename, e.g.
`statquest-r-squared.yaml`). Pages list the ids they show in their own `resources.yaml`:

```yaml
# concepts/r-squared/resources.yaml
- khan-r-squared-intuition
- statquest-r-squared
```

Unknown or duplicate ids are errors; a resource file nobody references is a warning. A shared
resource has one `why`, so write it to fit every page that lists it.

```yaml
# resources/statquest-r-squared.yaml
type: video                   # blog | video | paper | interactive | course
title: "R-squared, Clearly Explained!!!"
url: https://www.youtube.com/watch?v=2AQKmw14mHM
author: StatQuest with Josh Starmer
level: beginner               # beginner | intermediate | advanced
duration_min: 11
why: "One line on why this is worth the learner's time."
source: curated               # curated | own
checked: 2026-10-20           # last link check
```

## Concept code

`code/<library>.py` implements the concept from scratch, one file per library in
`CodeLibrary` (`numpy`, `pytorch`). Each file:

- runs on its own: seeded random data, no files, no network;
- prints what to look at, and `assert`s its result (e.g. against `np.linalg.lstsq`), so a
  passing run means the code is right;
- stays short (≈ 20–45 lines) and comments say *why*.

Explain blocks exactly like a pack's templates (ADR 0006): put `# @a:<anchor>` above a block
and give it an entry in `code/annotations.yaml`, under the library's section. Anchors and
entries must match 1:1; `concepts` link to other concept pages. Identical entries can be
shared with a YAML alias:

```yaml
numpy:
  solve: &solve
    span: block            # up to the next anchor; default `line`
    short: "One sentence."
    deep: |
      Markdown with $math$.
    concepts: [mean-squared-error]
pytorch:
  solve: *solve
```

Tests run every NumPy file. PyTorch files are compile-checked, and run too wherever torch is
installed (it isn't a project dependency). A missing library is a backlog gap. Pages show the
files as `<concept_id>_<library>.py`: a file named `numpy.py` would shadow NumPy when run.

## Rules

- Every concept needs at least one beginner blog/post and one video; every algorithm needs a
  complete overview. Gaps go to the generated content backlog.
- Curation rubric: correct, clear, recent enough for current library APIs, no paywall, matches the
  stated level. One excellent resource beats five average ones.
- Own videos live on YouTube and are referenced like any other video.
