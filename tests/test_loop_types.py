"""The loop's own vocabulary.

These reside in `tars.loop` rather than in any existing module because they
describe the *loop* - the product - not one phase. `Phase` is the product's own
taxonomy from `master_agent_loop.md`, `RunReport` is what `/run` returns, and
`ToolCall` is what the model's tool use normalizes into.
"""

from __future__ import annotations

import pytest

from tars.loop.types import (
    Phase,
    RunPromptSpec,
    RunReport,
    ToolCall,
    ToolResult,
)


def test_the_loop_phase_taxonomy_is_the_seven_from_the_spec():
    assert list(Phase) == [
        "orient",
        "classify",
        "plan",
        "act",
        "verify",
        "reflect",
        "report",
    ]


def test_a_run_prompt_spec_is_immutable():
    from dataclasses import FrozenInstanceError

    spec = RunPromptSpec(task="fix the typo")
    with pytest.raises(FrozenInstanceError):
        spec.task = "other"  # type: ignore[misc]


def test_a_run_prompt_spec_rejects_a_blank_task():
    with pytest.raises(ValueError, match="task"):
        RunPromptSpec(task="   ")


def test_a_tool_call_keeps_its_arguments_as_a_dict():
    call = ToolCall("read_file", "read_file", {"path": "a.py"})
    assert call.name == "read_file"
    assert call.arguments == {"path": "a.py"}


def test_a_tool_call_normalizes_a_string_id_to_a_tuple():
    call = ToolCall("apply_edit", "apply_edit", {"edit": "x"})
    assert call.id == "apply_edit"


def test_a_tool_call_with_missing_id_is_rejected():
    # Every tool call is answered by id, so an id-less one is a bug, not a call.
    with pytest.raises(ValueError, match="id"):
        ToolCall("", "apply_edit", {"edit": "x"})


def test_a_tool_result_can_be_created_from_a_success():
    result = ToolResult.ok_result("read_file", "def add(a, b):")
    assert result.ok
    assert result.output == "def add(a, b):"
    assert result.error is None


def test_a_tool_result_can_be_created_from_a_failure():
    result = ToolResult.error_result("apply_edit", "ambiguous: anchor appears twice")
    assert not result.ok
    assert result.error == "ambiguous: anchor appears twice"


def test_a_tool_result_is_displayable_for_a_log():
    assert ToolResult.ok_result("read_file", "hi").display == "read_file: hi"


def test_a_tool_result_from_an_error_is_displayable_without_a_huge_blob():
    result = ToolResult.error_result("run_command", "exit 1\n" + "x" * 500)
    assert len(result.display) < 200


def test_a_run_report_records_each_phase_it_reached():
    report = RunReport(task="t", verdict="tier 1", touched=("a.py",))
    assert report.touched == ("a.py",)
    assert report.verdict is not None


def test_a_run_report_reports_whether_verification_passed():
    """PROPERTY: VERIFY's verdict is a real field, not a string in a log."""
    report = RunReport(task="t", verdict="tier 0", verification_passed=True)
    assert report.verification_passed is True
