"""One Session, alive across commands.

The REPL's whole reason to be persistent is that state carries: `/route` records
a verdict `/run` can reuse, `/run` grows a transcript `/memory` can reflect on,
and `/verify` leaves a result `/status` can report. A Session that lost any of
that would make the REPL a slower way to run four separate CLIs.
"""

from __future__ import annotations

from pathlib import Path

from tars.loop.session import Session, default_model, default_tier0_model
from tars.providers import Message, ToolCall


def test_a_session_defaults_to_the_current_directory():
    assert Session().root == Path()


def test_a_session_remembers_its_root(tmp_path: Path):
    assert Session(root=tmp_path).root == tmp_path


def test_a_new_session_has_no_transcript_yet():
    assert Session().messages == ()


def test_a_session_grows_its_transcript_in_order():
    session = Session()
    session.record(Message.user("fix the typo"))
    session.record(Message.assistant("done"))
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.messages[0].content == "fix the typo"


def test_the_transcript_is_exposed_as_a_tuple_so_a_caller_cannot_mutate_it():
    session = Session()
    session.record(Message.user("hi"))
    assert isinstance(session.messages, tuple)


def test_a_session_transcript_is_readable_by_the_memory_writer():
    """The REFLECT seam. If this drifts, /memory silently reflects on nothing."""
    from tars.memory import parse_messages

    session = Session()
    session.record(Message.user("should we use stdlib or httpx?"))
    session.record(
        Message.assistant("We decided to use stdlib to stay dependency-free.")
    )

    transcript = parse_messages(session.as_messages(), session="repl")

    assert len(transcript.turns) == 2
    assert transcript.turns[1].role == "assistant"
    assert "stdlib" in transcript.turns[1].text


def test_a_session_records_the_last_router_verdict():
    from tars.router import classify

    session = Session()
    assert session.last_classification is None
    session.last_classification = classify("Fix the typo in the README.")
    assert session.last_classification.tier == 0


def test_a_session_records_the_last_gate_result():
    session = Session()
    assert session.last_verdict is None
    session.last_verdict = "pass"
    assert session.last_verdict == "pass"


def test_a_session_can_be_cleared_without_losing_its_root(tmp_path: Path):
    session = Session(root=tmp_path, model="agentrouter:claude-opus-5")
    session.record(Message.user("hi"))
    session.last_verdict = "fail"

    session.clear()

    assert session.messages == ()
    assert session.last_verdict is None
    assert session.root == tmp_path
    assert session.model == "agentrouter:claude-opus-5"


def test_a_persisted_clear_does_not_restore_old_messages(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(root=tmp_path, session_id="saved", store=store)
    session.record(Message.user("old context"))
    session.clear()
    assert Session.load(store, "saved").messages == ()


def test_persistence_round_trips_full_provider_message_shape(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(root=tmp_path, session_id="saved", store=store)
    session.record(
        Message.assistant(
            "inspect",
            tool_calls=(ToolCall("call-1", "read_file", {"path": "a.py"}),),
        ),
        Message.tool_result("call-1", "denied", is_error=True),
    )
    loaded = Session.load(store, "saved")
    assert loaded.messages == session.messages


# ── the model default is machine-dependent, and must say so ──────────────


def test_the_default_model_uses_agentrouter_when_that_is_what_is_configured():
    """A bare claude-opus-5 routes to api.anthropic.com, which has no key here."""
    model = default_model(
        {
            "ANTHROPIC_AUTH_TOKEN": "sk-x",
            "ANTHROPIC_BASE_URL": "https://agentrouter.org",
        }
    )
    assert model == "agentrouter:claude-opus-5"


def test_the_default_model_is_plain_anthropic_when_a_real_api_key_is_present():
    assert default_model({"ANTHROPIC_API_KEY": "sk-ant-real"}) == "claude-opus-5"


def test_an_explicit_env_override_wins():
    assert default_model({"TARS_MODEL": "gpt-5"}) == "gpt-5"


def test_with_no_credentials_at_all_the_default_is_still_nameable():
    """It must not crash at import time on a machine with nothing configured."""
    assert default_model({}) == "claude-opus-5"


def test_an_auth_token_without_a_base_url_is_not_assumed_to_be_agentrouter():
    """ANTHROPIC_AUTH_TOKEN alone could be a direct Anthropic bearer token."""
    assert default_model({"ANTHROPIC_AUTH_TOKEN": "sk-x"}) == "claude-opus-5"


def test_tier_zero_model_is_unset_unless_configured() -> None:
    assert default_tier0_model({}) is None
    assert (
        default_tier0_model({"TARS_TIER0_MODEL": "openrouter:test-fixture/free"})
        == "openrouter:test-fixture/free"
    )


def test_tier_model_map_defaults_higher_tiers_to_session_model() -> None:
    session = Session(model="flagship", tier_models={"0": None})
    assert session.tier_models == {"0": None, "1": "flagship", "2": "flagship"}


def test_explicit_tier_zero_config_beats_environment(monkeypatch) -> None:
    monkeypatch.setenv("TARS_TIER0_MODEL", "openrouter:env/free")
    session = Session(model="flagship", tier_models={"0": "openrouter:config/free"})
    assert session.tier_models["0"] == "openrouter:config/free"
    assert session.tier0_source == "config"


def test_environment_tier_zero_source_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("TARS_TIER0_MODEL", "openrouter:env/free")
    session = Session(model="flagship")
    assert session.tier_models["0"] == "openrouter:env/free"
    assert session.tier0_source == "env"


def test_tier_model_map_round_trips_through_config_event(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(
        root=tmp_path,
        model="flagship",
        tier_models={"0": "openrouter:test-fixture/free"},
        session_id="saved",
        store=store,
    )
    loaded = Session.load(store, "saved")
    assert loaded.tier_models == session.tier_models


def test_mode_and_pending_plan_round_trip_through_session_store(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore
    from tars.loop.types import InteractionMode

    store = SessionStore(tmp_path / "sessions")
    session = Session(root=tmp_path, session_id="modes", store=store)
    session.set_mode(InteractionMode.PLAN)
    session.save_plan("fix it", "1. Inspect the file.\n2. Run tests.")

    loaded = Session.load(store, "modes")
    assert loaded.mode is InteractionMode.PLAN
    assert loaded.pending_plan is not None
    assert loaded.pending_plan.task == "fix it"


def test_clear_removes_a_pending_plan_and_persists_that_fact(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(root=tmp_path, session_id="clear-mode", store=store)
    session.save_plan("task", "plan")
    session.clear()
    loaded = Session.load(store, "clear-mode")
    assert loaded.pending_plan is None


def test_loading_a_session_from_before_modes_defaults_to_chat(tmp_path: Path) -> None:
    from tars.loop.session_store import SessionStore
    from tars.loop.types import InteractionMode

    store = SessionStore(tmp_path / "sessions")
    store.append(
        "legacy",
        "session_started",
        {"root": str(tmp_path), "model": "fake", "auto_confirm": False},
    )
    assert Session.load(store, "legacy").mode is InteractionMode.CHAT
