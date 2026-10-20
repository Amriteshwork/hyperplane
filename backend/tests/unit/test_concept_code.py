"""Every concept implementation in content/ runs and passes its own asserts."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from app.content.catalog import ContentCatalog
from app.content.models import CodeLibrary, code_module

CONTENT = Path(__file__).parents[3] / "content"
CATALOG = ContentCatalog.load(CONTENT)
CODE = [
    pytest.param(
        code_module(concept.id, code.library), code.library, id=f"{concept.id}-{code.library}"
    )
    for concept in CATALOG.concepts()
    for code in concept.code
]
# Torch is not a project dependency; its snippets are compiled always and run where it's installed.
RUNNABLE = {
    CodeLibrary.NUMPY: True,
    CodeLibrary.PYTORCH: importlib.util.find_spec("torch") is not None,
}


@pytest.fixture(scope="module")
def code_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every concept file under the name pages show, so imports between concepts resolve."""
    folder = tmp_path_factory.mktemp("concept_code")
    for concept in CATALOG.concepts():
        for code in concept.code:
            (folder / f"{code_module(concept.id, code.library)}.py").write_text(code.source)
    return folder


@pytest.mark.parametrize(("module", "library"), CODE)
def test_concept_code_runs(code_dir: Path, module: str, library: CodeLibrary) -> None:
    path = code_dir / f"{module}.py"
    compile(path.read_text(), path.name, "exec")
    if not RUNNABLE[library]:
        pytest.skip(f"{library} is not installed")
    # -E -s: ignore PYTHON* variables and user site-packages; the script's folder holds the
    # other concepts' files, as it would for a learner who downloaded them together.
    result = subprocess.run(
        [sys.executable, "-E", "-s", path.name],
        cwd=code_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_imported_concept_code_runs_nothing_on_import(code_dir: Path) -> None:
    """Importing a shared file (e.g. `mse`) must not print its demo."""
    shared = sorted(
        code_module(use.concept, code.library)
        for concept in CATALOG.concepts()
        for code in concept.code
        for use in code.uses
        if RUNNABLE[code.library]
    )
    assert shared, "no concept imports another's code yet"
    script = "; ".join(f"import {module}" for module in dict.fromkeys(shared))
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-c", script],
        cwd=code_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
