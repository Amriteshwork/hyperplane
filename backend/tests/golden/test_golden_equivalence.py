"""Golden equivalence (ARCHITECTURE §4): the rendered `train.py` predicts what the platform run did.

For every pack and every matching sample dataset, the platform trains in-process, then the
script shown to the learner runs as a subprocess on the same data, and both models must give
the same predictions on the same held-out rows.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from app.ingest.parquet import read_parquet, to_parquet_bytes
from app.ml.base import AlgorithmPack
from app.ml.codegen import render
from app.ml.evaluation import EVALUATORS
from app.ml.preprocessing import FEATURE_ROLES, assign_roles
from app.ml.profiling import profile_dataframe
from app.ml.registry import AlgorithmRegistry
from app.ml.training import TrainingConfig, codegen_context, train
from app.ml.types import TaskType
from app.samples.catalog import SampleCatalog
from tests.conftest import DATASETS_DIR

REGISTRY = AlgorithmRegistry.discover()
SAMPLES = SampleCatalog.load(DATASETS_DIR)
# Extra (sample, target) pairs that exercise missing values and categoricals for a task type.
EXTRA_TARGETS = {TaskType.REGRESSION: [("penguins", "body_mass_g")]}


def _cases() -> list[Any]:
    cases = []
    for pack in REGISTRY:
        for task in sorted(pack.spec.task_types & EVALUATORS.keys()):
            targets = [(s.meta.id, s.meta.target) for s in SAMPLES.list() if s.meta.task == task]
            for sample_id, target in targets + EXTRA_TARGETS.get(task, []):
                cases.append(
                    pytest.param(pack, sample_id, target, id=f"{pack.spec.id}-{sample_id}")
                )
    return cases


@pytest.mark.parametrize(("pack", "sample_id", "target"), _cases())
def test_rendered_script_reproduces_the_platform_run(
    pack: AlgorithmPack, sample_id: str, target: str, tmp_path: Path
) -> None:
    sample = SAMPLES.get(sample_id)
    if sample.status != "available":
        pytest.skip(f"{sample_id} not fetched (run `make datasets`)")
    # The platform reads the stored Parquet; the learner gets it back as CSV.
    df = read_parquet(to_parquet_bytes(pd.read_csv(sample.path)))
    df.to_csv(tmp_path / "data.csv", index=False)

    profile = profile_dataframe(df)
    usable = [
        c.name
        for c in profile.columns
        if c.name != target and FEATURE_ROLES[c.semantic_type] is not None
    ]
    config = TrainingConfig(
        target=target,
        task_type=next(c.task_type for c in profile.target_candidates if c.column == target),
        preprocessing=assign_roles(profile, usable),
        test_size=0.2,
        seed=42,
        params=pack.params_model().model_dump(mode="json"),
    )
    platform = train(df, config, pack)

    code = render(pack.templates()["train"], codegen_context(config, "data.csv")).code
    (tmp_path / "train.py").write_text(code)
    script = subprocess.run(
        [sys.executable, "train.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert script.returncode == 0, script.stderr

    script_model = joblib.load(tmp_path / "model.joblib")  # written by our own script
    np.testing.assert_allclose(
        script_model.predict(platform.X_test), platform.predictions, rtol=1e-9, atol=1e-9
    )
    first_metric = platform.report.evaluation.metrics[0]
    printed = re.search(
        rf"^{re.escape(first_metric.label)}:\s+(-?[\d.]+)$", script.stdout, re.MULTILINE
    )
    assert printed is not None, script.stdout
    assert float(printed[1]) == pytest.approx(first_metric.value, abs=5e-5)
