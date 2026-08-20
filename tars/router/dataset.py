"""The labeled task set, and the two splits it is read through.

The dataset is data, not code: `data/labeled_tasks.json` carries the cases and
the rubric that produced them, side by side, so a label can be argued with
without reading Python. Every case states *why* it got its labels — a label with
no rationale cannot be corrected, only overruled.

The dev/holdout split is recorded per case rather than computed. A split derived
from a hash or an index silently reshuffles the moment a case is inserted, which
turns "accuracy went up" into an unanswerable question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

Tier = Literal[0, 1, 2]
Blast = Literal["low", "medium", "high"]
Split = Literal["dev", "holdout"]

DATA_PACKAGE = "tars.router"
DATA_PATH = ("data", "labeled_tasks.json")

VALID_TIERS: tuple[int, ...] = (0, 1, 2)
VALID_BLAST: tuple[str, ...] = ("low", "medium", "high")
VALID_SPLITS: tuple[str, ...] = ("dev", "holdout")


@dataclass(frozen=True)
class LabeledTask:
    """One hand-labeled task description.

    `reversible` is about the world, not the diff: a change that rewrites stored
    data is irreversible even when `git revert` restores the code that did it.
    """

    id: str
    task: str
    tier: int
    blast: str
    reversible: bool
    why: str
    split: str = "dev"


class DatasetError(Exception):
    """The shipped dataset is malformed.

    Its own tests would catch this, but the loader refuses too: a benchmark that
    silently skips unparseable rows reports an accuracy over a set nobody chose.
    """


def _raw() -> dict[str, Any]:
    path = files(DATA_PACKAGE)
    for part in DATA_PATH:
        path = path / part
    loaded = json.loads(path.read_text("utf-8"))
    if not isinstance(loaded, dict):
        raise DatasetError("labeled_tasks.json must hold a JSON object")
    return loaded


@lru_cache(maxsize=1)
def rubric() -> dict[str, Any]:
    """The labeling rules, as shipped next to the labels."""
    got = _raw().get("rubric")
    if not isinstance(got, dict):
        raise DatasetError("labeled_tasks.json is missing its 'rubric' object")
    return got


@lru_cache(maxsize=1)
def load_dataset() -> tuple[LabeledTask, ...]:
    """Every labeled case, in file order."""
    cases = _raw().get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetError("labeled_tasks.json has no 'cases' list")
    return tuple(_to_case(entry, index) for index, entry in enumerate(cases))


def _to_case(entry: Any, index: int) -> LabeledTask:
    if not isinstance(entry, dict):
        raise DatasetError(f"case {index} is not an object")
    missing = {"id", "task", "tier", "blast", "reversible", "why"} - set(entry)
    if missing:
        raise DatasetError(f"case {index} is missing {sorted(missing)}")

    case = LabeledTask(
        id=str(entry["id"]),
        task=str(entry["task"]),
        tier=int(entry["tier"]),
        blast=str(entry["blast"]),
        reversible=bool(entry["reversible"]),
        why=str(entry["why"]),
        split=str(entry.get("split", "dev")),
    )
    if case.tier not in VALID_TIERS:
        raise DatasetError(
            f"{case.id} has tier {case.tier}, expected one of {VALID_TIERS}"
        )
    if case.blast not in VALID_BLAST:
        raise DatasetError(
            f"{case.id} has blast {case.blast!r}, expected one of {VALID_BLAST}"
        )
    if case.split not in VALID_SPLITS:
        raise DatasetError(
            f"{case.id} has split {case.split!r}, expected one of {VALID_SPLITS}"
        )
    return case


def split(name: str) -> tuple[LabeledTask, ...]:
    """The cases assigned to one split."""
    if name not in VALID_SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {VALID_SPLITS}")
    return tuple(case for case in load_dataset() if case.split == name)


def dev_split() -> tuple[LabeledTask, ...]:
    """Cases the classifier's rules were allowed to be tuned against."""
    return split("dev")


def holdout_split() -> tuple[LabeledTask, ...]:
    """Cases held back, so a reported number means something."""
    return split("holdout")
