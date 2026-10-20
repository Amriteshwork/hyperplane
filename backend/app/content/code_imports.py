"""Imports between concept code files (ADR 0016), checked by parsing: content code never runs here.

A concept's code may `from <prerequisite>_<library> import name` (e.g. `from
normal_equation_numpy import least_squares`) when the prerequisite is a transitive one, in the
same library, defines `name` at its top level, and keeps its demo under `if __name__ ==
"__main__":` so importing it runs nothing.
"""

import ast
from collections.abc import Collection, Mapping

from app.content.graph import ConceptGraphError
from app.content.models import CodeLibrary, CodeUse, code_module

#: Top-level statements that are safe to run on import.
_DEFINITIONS = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.TypeAlias,
)


def code_uses(
    filename: str,
    source: str,
    library: CodeLibrary,
    ancestors: Collection[str],
    sources: Mapping[str, Mapping[CodeLibrary, str]],
) -> tuple[CodeUse, ...]:
    """What `source` imports from other concepts' code. `sources` holds every concept's code."""
    modules = {code_module(c, lib): (c, lib) for c in sources for lib in CodeLibrary}
    uses: dict[str, list[str]] = {}
    for node in ast.walk(_parse(filename, source)):
        if isinstance(node, ast.Import) and any(a.name in modules for a in node.names):
            raise ConceptGraphError(
                f"{filename}:{node.lineno}: write `from <module> import <name>` so the page can "
                "show what is used"
            )
        if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in modules):
            continue
        concept, imported_library = modules[node.module]
        where = f"{filename}:{node.lineno}: imports {node.module}"
        if imported_library is not library:
            raise ConceptGraphError(f"{where}, but code only imports code in its own library")
        if concept not in ancestors:
            raise ConceptGraphError(f"{where}, but {concept!r} is not a prerequisite")
        target = sources[concept].get(library)
        if target is None:
            raise ConceptGraphError(f"{where}, but {concept!r} has no {library} code")
        names = [alias.name for alias in node.names]
        if missing := sorted(set(names) - _importable_names(f"{node.module}.py", target)):
            raise ConceptGraphError(f"{where}, which defines no {missing}")
        uses.setdefault(concept, []).extend(names)
    return tuple(CodeUse(concept=c, names=tuple(dict.fromkeys(n))) for c, n in uses.items())


def _importable_names(filename: str, source: str) -> set[str]:
    """Names defined at the top level; fails if importing would run more than definitions."""
    names: set[str] = set()
    for node in _parse(filename, source).body:
        if _is_docstring(node) or _is_main_guard(node):
            continue
        if not isinstance(node, _DEFINITIONS):
            raise ConceptGraphError(
                f"{filename}:{node.lineno}: other concepts import this file, so code outside "
                'definitions must go under `if __name__ == "__main__":`'
            )
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _parse(filename: str, source: str) -> ast.Module:
    try:
        return ast.parse(source, filename)
    except SyntaxError as exc:
        raise ConceptGraphError(f"{filename}:{exc.lineno}: {exc.msg}") from exc


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_main_guard(node: ast.stmt) -> bool:
    """`if __name__ == "__main__":` with no `else`."""
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )
