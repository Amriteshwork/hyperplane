"""Every concept implementation in content/ runs and passes its own asserts."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from app.content.catalog import ContentCatalog
from app.content.models import CodeLibrary

CONTENT = Path(__file__).parents[3] / "content"
CODE = [
    pytest.param(concept.id, code.library, code.source, id=f"{concept.id}-{code.library}")
    for concept in ContentCatalog.load(CONTENT).concepts()
    for code in concept.code
]
# Torch is not a project dependency; its snippets are compiled always and run where it's installed.
RUNNABLE = {
    CodeLibrary.NUMPY: True,
    CodeLibrary.PYTORCH: importlib.util.find_spec("torch") is not None,
}


@pytest.mark.parametrize(("concept_id", "library", "code"), CODE)
def test_concept_code_runs(concept_id: str, library: CodeLibrary, code: str) -> None:
    compile(code, f"{concept_id}/{library}.py", "exec")
    if not RUNNABLE[library]:
        pytest.skip(f"{library} is not installed")
    # -I: isolated, so nothing local (like a file named numpy.py) shadows the library.
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
