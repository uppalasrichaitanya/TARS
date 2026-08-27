"""The optional inline composer keeps mode cycling deterministic."""

from tars.loop.tui import next_mode
from tars.loop.types import InteractionMode


def test_tab_mode_cycle_is_chat_plan_agent() -> None:
    assert next_mode(InteractionMode.CHAT) is InteractionMode.PLAN
    assert next_mode(InteractionMode.PLAN) is InteractionMode.AGENT
    assert next_mode(InteractionMode.AGENT) is InteractionMode.CHAT


def test_composer_can_follow_a_resumed_session() -> None:
    from tars.loop.session import Session
    from tars.loop.tui import Composer

    composer = object.__new__(Composer)
    composer._session = Session(model="old")
    resumed = Session(model="resumed", mode=InteractionMode.PLAN)
    composer.set_session(resumed)
    assert composer._session is resumed
