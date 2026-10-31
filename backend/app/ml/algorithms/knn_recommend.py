"""Suitability rules shared by the KNN packs: how the data's size and shape suit distances."""

from app.ml.profiling import DatasetProfile, SemanticType

_MODELLED = {SemanticType.NUMERIC, SemanticType.BOOLEAN, SemanticType.CATEGORICAL}
#: Above this many feature columns, nearest and farthest rows start to look alike.
_MANY_FEATURES = 20
#: Above this many rows, every prediction searching them starts to feel slow.
_MANY_ROWS = 50_000
#: Below this many rows per feature, the neighbourhoods are too sparse to mean much.
_SPARSE_ROWS_PER_FEATURE = 10


def data_fit(profile: DatasetProfile) -> tuple[float, list[str]]:
    """A starting score and the reasons behind it; the packs add what depends on the task."""
    usable = [c for c in profile.columns if c.semantic_type in _MODELLED]
    categorical = [c for c in usable if c.semantic_type is not SemanticType.NUMERIC]
    # The profile includes the target column; each category becomes one one-hot column.
    n_features = max(len(usable) - len(categorical) - 1, 0) + sum(c.unique for c in categorical)
    value = 0.6
    reasons = []

    if n_features > _MANY_FEATURES:
        value -= 0.2
        reasons.append(
            f"About {n_features} features after encoding: in that many dimensions the nearest "
            "and farthest rows are almost equally far (the curse of dimensionality). Drop "
            "uninformative columns first; every one of them adds noise to each distance."
        )
    elif profile.n_rows / max(n_features, 1) < _SPARSE_ROWS_PER_FEATURE:
        value -= 0.1
        reasons.append(
            f"Only ~{profile.n_rows / max(n_features, 1):.0f} rows per feature: neighbours will "
            "be far away, so they say little about the query."
        )
    else:
        value += 0.1
        reasons.append(
            f"About {n_features} features and {profile.n_rows:,} rows: few enough dimensions "
            "for 'nearby' to mean something, and enough rows to find close neighbours."
        )

    if profile.n_rows > _MANY_ROWS:
        value -= 0.2
        reasons.append(
            f"{profile.n_rows:,} rows: training is instant, but the model keeps every row and "
            "searches them for each prediction, so predictions will be slow and the saved "
            "model large."
        )

    if categorical:
        reasons.append(
            f"{len(categorical)} categorical column(s) become one-hot columns: two rows differ "
            "by the same distance whichever categories they hold, which may not be what "
            "similar means here."
        )
    missing = [c.name for c in usable if c.missing]
    if missing:
        reasons.append(f"{len(missing)} column(s) have missing values; they will be imputed first.")
    return value, reasons
