"""The prerequisite graph between concepts (ADR 0016): validation and learning paths."""

from collections.abc import Mapping, Sequence

type Prerequisites = Mapping[str, Sequence[str]]


class ConceptGraphError(ValueError):
    """Prerequisites or code imports break the concept graph's rules."""


def check_graph(prerequisites: Prerequisites) -> None:
    """Every prerequisite exists and none is circular."""
    for concept, required in prerequisites.items():
        if unknown := [p for p in required if p not in prerequisites]:
            raise ConceptGraphError(f"concepts/{concept}: unknown prerequisites {unknown}")

    done: set[str] = set()

    def visit(concept: str, trail: list[str]) -> None:
        if concept in done:
            return
        if concept in trail:
            cycle = [*trail[trail.index(concept) :], concept]
            raise ConceptGraphError(f"prerequisite cycle: {' → '.join(cycle)}")
        for prerequisite in prerequisites[concept]:
            visit(prerequisite, [*trail, concept])
        done.add(concept)

    for concept in sorted(prerequisites):
        visit(concept, [])


def learning_path(prerequisites: Prerequisites, concept_id: str) -> list[str]:
    """Every transitive prerequisite of `concept_id`, each after everything it needs.

    The concept itself is not included.
    """
    return reading_order(prerequisites, [concept_id])[:-1]


def reading_order(prerequisites: Prerequisites, concept_ids: Sequence[str]) -> list[str]:
    """The concepts and all their prerequisites, each once and after everything it needs.

    Depth-first in the order targets and prerequisites are written, so the path reads the
    way the author listed them.
    """
    path: list[str] = []

    def visit(concept: str) -> None:
        if concept in path:
            return
        for prerequisite in prerequisites[concept]:
            visit(prerequisite)
        path.append(concept)

    for concept in concept_ids:
        visit(concept)
    return path


def check_route(prerequisites: Prerequisites, goal: str, route: Sequence[str]) -> None:
    """`route` must hold `goal` and everything it builds on, once each, after what each needs."""
    if goal not in prerequisites:
        raise ConceptGraphError(f"unknown goal concept {goal!r}")
    needed = set(reading_order(prerequisites, [goal]))
    if missing := sorted(needed - set(route)):
        raise ConceptGraphError(f"route to {goal!r} is missing {missing}")
    if extra := sorted(set(route) - needed):
        raise ConceptGraphError(f"route to {goal!r} has concepts it doesn't need: {extra}")
    if len(set(route)) != len(route):
        raise ConceptGraphError(f"route to {goal!r} lists a concept twice")
    seen: set[str] = set()
    for concept in route:
        if early := [p for p in prerequisites[concept] if p not in seen]:
            raise ConceptGraphError(f"{concept!r} comes before its prerequisites {early}")
        seen.add(concept)
