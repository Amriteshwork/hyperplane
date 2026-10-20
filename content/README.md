# Content

Learning content is code: Markdown + YAML, reviewed in PRs, validated by
`uv run python -m scripts.content_check` (from `backend/`), and indexed by the API at startup.

```
content/
  algorithms/<algo_id>/        one folder per algorithm pack (id = pack id, e.g. linear_regression)
    overview.md                frontmatter + required sections (below)
    resources.yaml             ids of its curated resources, in display order
  concepts/<concept_id>/       one folder per concept (kebab-case, e.g. train-test-split)
    concept.md                 short explanation; frontmatter: title, summary, area, prerequisites[]
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

We link to the best existing explanations rather than rewriting them (ADR 0016). Each link
lives once in `resources/<resource_id>.yaml` (kebab-case id = filename, e.g.
`statquest-r-squared.yaml`). Pages list the links they show in their own `resources.yaml`,
each with a required `note`: what the learner gets from that link **on this page**:

```yaml
# concepts/r-squared/resources.yaml
- id: khan-r-squared-intuition
  note: "Builds R² from the variation around the mean and around the line, with a worked example."
- id: statquest-r-squared
  note: "Derives R² from 'variation around the mean vs around the fit', so the formula sticks."
```

Unknown or duplicate ids are errors; a resource file nobody references is a warning. A
shared resource has one `why` (why it's worth anyone's time); the page's `note` is specific.
Only describe what you've checked the resource covers.

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

- runs on seeded random data, no files, no network, needing only the files it imports;
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

## Concept graph (ADR 0016)

`area` places the concept on the concept index (`/concepts`): `foundations` (maths & NumPy),
`optimisation`, `evaluation` or `features`. A new area is a new `ConceptArea` value.

`prerequisites` in `concept.md` form a graph; loading fails on an unknown prerequisite or a
cycle. Each concept page shows its **learning path**: every prerequisite, transitively, in an
order where each comes after what it needs. Foundations (vectors, derivatives, probability…)
are ordinary concepts, so paths reach them.

### Reusing another concept's code

Build on earlier concepts instead of re-implementing them. Files are shown and downloaded as
`<concept_id>_<library>.py`, and import each other under those names:

```python
# @a:reuse
from mean_squared_error_numpy import mse
from normal_equation_numpy import least_squares
```

The loader checks every such import without running code:

- the imported concept must be a (transitive) prerequisite, in the same library;
- use `from … import name`, and `name` must be defined at the top level of that file;
- a file that others import must keep its demo under `if __name__ == "__main__":`, so
  importing it runs nothing (anchors inside that block work as usual).

The page lists what each file uses, with links, and offers "Download with dependencies": a
zip of the file, everything it imports (transitively) and a README with the run command. The
tests run every file next to the files it imports, and run a downloaded bundle.
