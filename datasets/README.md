# Built-in datasets

Sample datasets users can pick in the Lab instead of uploading. One folder per dataset:

```
datasets/<dataset_id>/         kebab-case id (must match `id` in meta.yaml)
  data.csv                     the table
  meta.yaml                    metadata (below), validated at API startup
```

Every `data.csv` is produced by `backend/scripts/sample_datasets.py`, so it can be rebuilt
exactly (the synthetic set uses a fixed seed).

- **Vendored** (`vendored: true`): the licence allows redistribution, so the CSV is committed.
- **Fetched** (`vendored: false`): no licence could be verified, so the CSV is git-ignored and
  downloaded instead. That happens when the Docker image is built, in CI, and locally with
  `make datasets`. The API lists a sample whose CSV is missing as `status: not-fetched`.

## `meta.yaml`

```yaml
id: synthetic-linear
name: Synthetic Linear
description: One line on what the data is and why it is a good teaching set.
task: regression               # regression | binary | multiclass
target: y                      # default target column
rows: 500
columns: 6
vendored: true
licence: CC0 1.0               # or "Unverified — see source" when vendored is false
source: Who published it / how it was obtained
url: https://…                 # or null
citation: "…"                  # or null
```

## Rules

- Only vendor a dataset whose licence allows redistribution, and record that licence here.
- Keep files small (≤ 10 MB) so the repo stays light. Larger sets are fetched.
- Every pack's contract tests train on each sample dataset of a supported task type, so adding a
  dataset here extends test coverage automatically.

Titanic is planned but not added yet, because no source with a clear licence has been found.
