"""The browser surface stays useful without Playwright being installed.

The serializer, parser, expectation matcher, and CLI grammar are the durable
parts of an accessibility-tree test.  They run hermetically; only the final
round-trip test in ``test_browser_live.py`` needs a browser process.
"""

from tars.browser import A11yNode, RefRegistry, parse_snapshot, serialize_tree


def fake_tree() -> A11yNode:
    return A11yNode(
        node_id="root",
        role="document",
        name="Account",
        children=(
            A11yNode(node_id="email", role="textbox", name="Email", value="a@b.test"),
            A11yNode(
                node_id="save",
                role="button",
                name="Save changes",
                states=("enabled",),
            ),
        ),
    )


def test_fake_tree_serializes_to_compact_text_and_parses_back() -> None:
    text = serialize_tree(fake_tree(), RefRegistry())
    assert isinstance(text, str)
    assert not isinstance(text, bytes)
    assert text.splitlines() == [
        '[ref=1] document "Account"',
        '  [ref=2] textbox "Email" value="a@b.test"',
        '  [ref=3] button "Save changes" enabled',
    ]

    parsed = parse_snapshot(text)
    assert parsed == (
        A11yNode(
            node_id="1",
            role="document",
            name="Account",
            children=(
                A11yNode(node_id="2", role="textbox", name="Email", value="a@b.test"),
                A11yNode(
                    node_id="3",
                    role="button",
                    name="Save changes",
                    states=("enabled",),
                ),
            ),
        ),
    )


def test_ref_ids_are_stable_across_snapshots_and_new_nodes_append() -> None:
    refs = RefRegistry()
    first = serialize_tree(fake_tree(), refs)
    changed = A11yNode(
        node_id="root",
        role="document",
        name="Account",
        children=(*fake_tree().children, A11yNode("cancel", "button", "Cancel")),
    )
    second = serialize_tree(changed, refs)

    assert '[ref=2] textbox "Email"' in first
    assert '[ref=2] textbox "Email"' in second
    assert '[ref=3] button "Save changes"' in second
    assert '[ref=4] button "Cancel"' in second


def test_long_values_and_whitespace_are_kept_cheap() -> None:
    node = A11yNode("status", "status", "  Build\n  finished  ", value="x" * 500)
    text = serialize_tree(node, RefRegistry(), max_value=20)
    assert '"Build finished"' in text
    assert "x" * 19 + "…" in text
    assert len(text) < 100
