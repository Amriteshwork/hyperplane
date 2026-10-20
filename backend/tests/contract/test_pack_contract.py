"""Shared contract every algorithm pack must pass (ARCHITECTURE §4). This suite is our LSP check.

Golden equivalence (rendered script == platform run) runs every pack in tests/golden.
"""

from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from app.content.catalog import ContentCatalog
from app.content.gaps import missing_sections
from app.core.config import get_settings
from app.ml.base import (
    AlgorithmPack,
    Capability,
    ReportsClassCoefficients,
    ReportsCoefficients,
    ReportsFeatureImportance,
)
from app.ml.codegen import CodegenContext, find_anchors, render
from app.ml.registry import AlgorithmRegistry
from app.samples.catalog import SampleCatalog
from tests.conftest import DATASETS_DIR

REGISTRY = AlgorithmRegistry.discover()
PACKS = [pytest.param(p, id=p.spec.id) for p in REGISTRY]
CONTENT_DIR = (Path(__file__).parents[1] / ".." / get_settings().content_dir).resolve()
CONTENT = ContentCatalog.load(CONTENT_DIR)
SAMPLES = SampleCatalog.load(DATASETS_DIR)


def test_at_least_one_pack_is_registered() -> None:
    assert len(REGISTRY) > 0


@pytest.mark.parametrize("pack", PACKS)
def test_spec_links_to_existing_content(pack: AlgorithmPack) -> None:
    content = CONTENT.algorithm(pack.spec.id)
    assert pack.spec.content_path == f"content/algorithms/{pack.spec.id}"
    assert missing_sections(content.body) == []
    assert CONTENT.family_of(pack.spec.id) is not None
    assert [c for c in pack.spec.concepts if not CONTENT.has_concept(c)] == []


@pytest.mark.parametrize("pack", PACKS)
def test_declared_capabilities_match_protocols(pack: AlgorithmPack) -> None:
    supports = pack.spec.supports
    assert (Capability.COEFFICIENTS in supports) == isinstance(pack, ReportsCoefficients)
    assert (Capability.CLASS_COEFFICIENTS in supports) == isinstance(pack, ReportsClassCoefficients)
    assert (Capability.FEATURE_IMPORTANCE in supports) == isinstance(pack, ReportsFeatureImportance)


@pytest.mark.parametrize("pack", PACKS)
def test_params_schema_is_described_and_levelled(pack: AlgorithmPack) -> None:
    schema = pack.params_model.model_json_schema()
    for name, field in schema.get("properties", {}).items():
        assert field.get("description"), f"{name} needs a description (shown as help text)"
        assert field.get("level") in {"beginner", "advanced"}, f"{name} needs a level"
    pack.params_model()  # every parameter has a default


@pytest.mark.parametrize("pack", PACKS)
def test_defaults_train_on_every_matching_sample(pack: AlgorithmPack) -> None:
    samples = [
        s for s in SAMPLES.list() if s.meta.task in pack.spec.task_types and s.status == "available"
    ]
    if not samples:
        pytest.skip("no matching sample datasets fetched (run `make datasets`)")
    for sample in samples:
        df = pd.read_csv(sample.path)
        X = df.drop(columns=sample.meta.target).select_dtypes("number")
        y = df[sample.meta.target]
        model = make_pipeline(SimpleImputer(), pack.build_estimator(pack.params_model()))
        model.fit(X, y)
        assert len(model.predict(X.head())) == 5


@pytest.mark.parametrize("pack", PACKS)
def test_anchors_and_annotations_match(pack: AlgorithmPack) -> None:
    templates = pack.templates()
    assert {"train", "scratch"} <= set(templates)
    # A template may branch on the task type; every branch must keep the anchors in sync.
    contexts = [
        CodegenContext(
            data_file="data.csv",
            target="y",
            task_type=task,
            numeric_features=["a", "b"],
            categorical_features=["c"],
            test_size=0.2,
            seed=0,
            params=pack.params_model().model_dump(mode="json"),
        )
        for task in sorted(pack.spec.task_types)
    ]
    for (name, template), context in product(templates.items(), contexts):
        anchors = find_anchors(template.source)
        assert sorted(anchors) == sorted(template.annotations), name
        rendered = render(template, context)  # raises on any mismatch after rendering
        assert "@a:" not in rendered.code
        for span in rendered.spans:
            assert span.annotation.short and span.annotation.deep, f"{name}:{span.anchor}"
            assert all(CONTENT.has_concept(c) for c in span.annotation.concepts)
        compile(rendered.code, template.filename, "exec")
