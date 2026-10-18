# Built-in datasets

Sample datasets users can pick in the Lab instead of uploading. One folder per dataset:

```
datasets/<dataset_id>/         kebab-case id, e.g. california-housing
  data.parquet                 the table (Parquet; small sets may use data.csv)
  meta.yaml                    metadata (below)
```

## `meta.yaml`

```yaml
name: California Housing
description: Median house values for California districts (1990 census).
task: regression               # regression | binary | multiclass
target: MedHouseVal            # default target column
rows: 20640
columns: 9
source: scikit-learn (sklearn.datasets.fetch_california_housing)
license: Public domain
url: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset
```

## Rules

- Licence must allow redistribution; record it in `meta.yaml`.
- Keep files small (≤ 10 MB) so the repo stays light; larger sets are downloaded by a script.
- Every pack's contract tests train on each sample dataset of a supported task type, so adding a
  dataset here extends test coverage automatically.

Planned for M0: Iris, Penguins, Titanic, Breast Cancer, Wine, California Housing, Diabetes, and a
small synthetic linear set.
