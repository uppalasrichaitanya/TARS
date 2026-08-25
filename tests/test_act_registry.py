"""Executor registration, and the laziness that makes an outside harness optional.

The registry exists so that ACT's backend is a name, not an import. That matters
for one specific reason: an external harness is a heavy dependency, and a repo
that only ever uses the built-in executor must never pay for it — not in import
time, not in a crash when it isn't installed.
"""

from __future__ import annotations

import pytest

from tars.act import registry
from tars.act.local import LocalExecutor
from tars.act.registry import (
    UnknownExecutorError,
    get_executor,
    install_default_executors,
    register_executor,
    reset_executors,
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_executors()
    yield
    reset_executors()
    install_default_executors()


def test_the_local_executor_is_registered_by_default():
    install_default_executors()
    assert isinstance(get_executor("local"), LocalExecutor)


def test_an_unknown_name_is_an_error_that_lists_what_is_available():
    install_default_executors()
    with pytest.raises(UnknownExecutorError, match="local"):
        get_executor("nonexistent")


def test_a_registered_factory_is_constructed_once_and_reused():
    calls = []

    def factory():
        calls.append(1)
        return LocalExecutor(".")

    register_executor("counted", factory)
    first = get_executor("counted")
    second = get_executor("counted")

    assert first is second
    assert len(calls) == 1


def test_registering_a_factory_does_not_call_it():
    """The whole point of the seam.

    An external harness's adapter must not be imported, and its dependencies must
    not be required, merely because the name exists.
    """

    def explode():
        raise AssertionError("a registered factory was constructed eagerly")

    register_executor("landmine", explode)  # must not raise


def test_re_registering_a_name_replaces_the_cached_instance():
    register_executor("swappable", lambda: LocalExecutor("first"))
    first = get_executor("swappable")
    register_executor("swappable", lambda: LocalExecutor("second"))
    second = get_executor("swappable")

    assert first is not second
    assert str(second.root) == "second"


def test_resetting_drops_cached_instances_but_keeps_registrations():
    register_executor("keeper", lambda: LocalExecutor("."))
    first = get_executor("keeper")
    reset_executors()
    assert get_executor("keeper") is not first


def test_installing_defaults_does_not_clobber_a_caller_override():
    """A project that has deliberately swapped `local` keeps its choice."""
    sentinel = LocalExecutor("overridden")
    register_executor("local", lambda: sentinel)
    install_default_executors()
    assert get_executor("local") is sentinel


def test_the_known_names_are_reportable():
    install_default_executors()
    assert "local" in registry.known_executors()
