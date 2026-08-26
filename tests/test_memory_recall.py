"""Lexical related recall crosses domains without weakening memory's trust rules."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tars import memory
from tars.memory.store import MemoryStore
from tars.memory.types import Candidate, Evidence, MemoryEntry, ReflectionReport

ROOT = Path(__file__).resolve().parent.parent
STAMP = "2026-08-26T12:00:00+00:00"


def candidate(
    text: str,
    *,
    domain: str,
    confidence: float = 0.9,
    supersedes: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        text=text,
        kind="decision",
        domains=(domain,),
        why="A later session should apply the established behavior.",
        confidence=confidence,
        evidence=(Evidence(turn=1, role="user", quote=text),),
        supersedes=supersedes,
    )


def remember(store: MemoryStore, *candidates: Candidate) -> None:
    store.write(
        ReflectionReport(session="related-recall", candidates=candidates),
        created_at=STAMP,
    )


def test_related_recall_ranks_shared_vocabulary_across_domains(tmp_path) -> None:
    store = MemoryStore(root=tmp_path)
    related = candidate(
        "Retry transient provider failures once before reporting interruption.",
        domain="providers",
        confidence=0.8,
    )
    unrelated = candidate(
        "Provider README examples are copied verbatim from source files.",
        domain="readme",
        confidence=0.99,
    )
    remember(store, unrelated, related)

    found = store.recall_related("Handle a transient provider failure with a retry")

    assert [entry.id for entry in found] == [
        MemoryEntry.from_candidate(related).id,
        MemoryEntry.from_candidate(unrelated).id,
    ]


def test_related_recall_breaks_similarity_ties_by_confidence(tmp_path) -> None:
    store = MemoryStore(root=tmp_path)
    lower = candidate("Retry transient failures.", domain="providers", confidence=0.8)
    higher = candidate(
        "Retry transient failures!", domain="verification", confidence=0.95
    )
    remember(store, lower, higher)

    found = store.recall_related("retry transient failures")

    assert [entry.confidence for entry in found] == [0.95, 0.8]


def test_related_recall_never_returns_pending_or_superseded_entries(tmp_path) -> None:
    store = MemoryStore(root=tmp_path)
    old = candidate("Retry transient failures twice.", domain="providers")
    pending = candidate(
        "Retry transient failures indefinitely.",
        domain="verification",
        confidence=0.5,
    )
    remember(store, old, pending)
    old_id = MemoryEntry.from_candidate(old).id
    replacement = candidate(
        "Retry transient failures once.",
        domain="loop",
        supersedes=(old_id,),
    )
    remember(store, replacement)

    found = store.recall_related("retry transient failures")

    assert [entry.id for entry in found] == [MemoryEntry.from_candidate(replacement).id]
    assert len(store.pending()) == 1


def test_related_recall_returns_nothing_for_empty_store_or_no_matches(tmp_path) -> None:
    store = MemoryStore(root=tmp_path)
    assert store.recall_related("retry transient failures") == ()

    remember(
        store,
        candidate("README examples stay verbatim.", domain="readme"),
    )
    assert store.recall_related("retry transient failures") == ()


def test_package_exposes_related_recall(tmp_path) -> None:
    store = MemoryStore(root=tmp_path)
    remember(
        store,
        candidate("Retry transient failures once.", domain="providers"),
    )

    assert memory.recall_related("transient retry", root=tmp_path)
    assert "recall_related" in memory.__all__


def test_similarity_module_has_only_standard_library_imports() -> None:
    path = ROOT / "tars" / "memory" / "similarity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
    )

    assert imports <= set(sys.stdlib_module_names)
