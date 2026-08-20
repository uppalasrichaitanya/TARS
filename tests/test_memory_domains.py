"""Domain tagging is the rule most likely to be violated by being helpful.

"Tag cross-cutting entries with every domain they touch (don't infer hidden
connections between domains — tag explicitly, or not at all)." So every tag here
has to be traceable to a marker that literally appears in the text, and matching
one domain must never pull in a second one that is merely *related* to it. A map
that knows the API serves the UI will tag a backend note as frontend too, and a
future ORIENT retrieving by domain then gets a fact that has nothing to do with
what it is working on.

Zero tags is a legitimate answer, and the schema already makes it a safe one: an
untagged candidate cannot auto-commit.

One safety property lives here too: domain names become filenames in the store,
so a name that can escape a directory is refused at the source.
"""

from __future__ import annotations

import json

import pytest

from tars.memory.domains import (
    DEFAULT_DOMAINS,
    Domain,
    DomainMap,
    load_domain_map,
)


def domains_for(text: str) -> tuple[str, ...]:
    return DEFAULT_DOMAINS.domains_for(text)


# ── matching on explicit markers ──────────────────────────────────────────


def test_a_path_in_the_text_tags_the_domain_that_owns_it() -> None:
    assert "providers" in domains_for("the retry lives in tars/providers/base.py")


def test_a_keyword_in_the_text_tags_its_domain() -> None:
    assert "infra" in domains_for("the Dockerfile pins python 3.12")


def test_text_with_no_marker_at_all_gets_no_tag() -> None:
    # Not a failure. An untagged candidate is one a human has to place, and the
    # schema already refuses to auto-commit it.
    assert domains_for("we agreed to keep the wording as it was") == ()


def test_matching_reports_the_marker_that_fired() -> None:
    # A tag nobody can trace is a tag nobody can correct.
    matches = DEFAULT_DOMAINS.match("the retry lives in tars/providers/base.py")
    assert [m.domain for m in matches] == ["providers"]
    assert matches[0].marker == "tars/providers"
    assert matches[0].kind == "path"


def test_a_statement_touching_two_domains_is_tagged_with_both() -> None:
    # Cross-domain is the common case, not an edge case — but only when both
    # domains are named outright.
    tags = domains_for("the pytest fixture spins up the Dockerfile image")
    assert set(tags) == {"testing", "infra"}


def test_one_domains_marker_never_drags_in_a_related_domain() -> None:
    # The rule this file exists for. A frontend note stays frontend even in a
    # project where the frontend obviously talks to the backend.
    assert domains_for("the React component reads the store") == ("frontend",)


def test_matching_is_case_insensitive() -> None:
    assert domains_for("the DOCKERFILE pins python") == domains_for(
        "the dockerfile pins"
    )


def test_a_keyword_inside_a_longer_word_does_not_match() -> None:
    # "protests" is not "tests"; substring matching would tag half the corpus.
    assert domains_for("the protests continued") == ()


def test_tags_come_back_sorted_so_two_runs_agree() -> None:
    assert domains_for("Dockerfile and pytest") == domains_for("pytest and Dockerfile")


def test_a_domain_is_reported_once_however_many_markers_hit() -> None:
    tags = domains_for("pytest, the tests/ directory, and another unit test")
    assert tags.count("testing") == 1


# ── the default map ───────────────────────────────────────────────────────


def test_the_default_map_covers_the_partitions_the_spec_names() -> None:
    for name in ("backend", "frontend", "infra", "data-model"):
        assert name in DEFAULT_DOMAINS.names


def test_the_default_map_covers_this_projects_own_partitions() -> None:
    # TARS is the first consumer; a map that can't tag TARS's own work is a map
    # that would send every entry from tonight to a human unplaced.
    for name in ("providers", "browser", "memory", "verification"):
        assert name in DEFAULT_DOMAINS.names


def test_every_default_domain_has_at_least_one_marker() -> None:
    for domain in DEFAULT_DOMAINS.domains:
        assert domain.paths or domain.keywords, f"{domain.name} can never match"


def test_every_default_domain_says_what_it_is_for() -> None:
    for domain in DEFAULT_DOMAINS.domains:
        assert domain.description.strip(), domain.name


# ── name safety ───────────────────────────────────────────────────────────


def test_a_domain_name_that_could_escape_a_directory_is_refused() -> None:
    # Domain names become filenames in the store. This is the only place that
    # can stop `../../etc/passwd` from being one.
    with pytest.raises(ValueError):
        Domain(name="../etc", description="bad", keywords=("x",))


def test_a_domain_name_with_a_separator_is_refused() -> None:
    with pytest.raises(ValueError):
        Domain(name="a/b", description="bad", keywords=("x",))


def test_a_domain_name_is_normalised_to_lower_case() -> None:
    assert (
        Domain(name="Data-Model", description="d", keywords=("x",)).name == "data-model"
    )


def test_an_empty_domain_name_is_refused() -> None:
    with pytest.raises(ValueError):
        Domain(name="  ", description="d", keywords=("x",))


def test_a_dashed_or_underscored_name_is_fine() -> None:
    assert (
        Domain(name="data_model", description="d", keywords=("x",)).name == "data_model"
    )


# ── project overrides ─────────────────────────────────────────────────────


def test_a_project_with_no_override_gets_the_default_map(tmp_path) -> None:
    assert load_domain_map(tmp_path).names == DEFAULT_DOMAINS.names


def test_a_project_can_declare_its_own_partitions(tmp_path) -> None:
    # A project's partitions are the project's call. The default map is a
    # starting point, not a taxonomy anyone else has to accept.
    path = tmp_path / ".tars" / "memory" / "domains.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "domains": {
                    "billing": {"description": "invoices", "keywords": ["invoice"]},
                    "ledger": {"description": "the ledger", "paths": ["src/ledger"]},
                }
            }
        )
    )
    loaded = load_domain_map(tmp_path)
    assert loaded.names == ("billing", "ledger")
    assert loaded.domains_for("the invoice total") == ("billing",)


def test_an_override_can_extend_the_defaults_instead_of_replacing_them(
    tmp_path,
) -> None:
    path = tmp_path / ".tars" / "memory" / "domains.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"extends": "default", "domains": {"billing": {"keywords": ["invoice"]}}}
        )
    )
    loaded = load_domain_map(tmp_path)
    assert "billing" in loaded.names
    assert "providers" in loaded.names


def test_a_malformed_override_says_which_file_is_wrong(tmp_path) -> None:
    path = tmp_path / ".tars" / "memory" / "domains.json"
    path.parent.mkdir(parents=True)
    path.write_text("{oops")
    with pytest.raises(ValueError, match="domains.json"):
        load_domain_map(tmp_path)


def test_an_override_with_an_unusable_domain_name_is_refused(tmp_path) -> None:
    path = tmp_path / ".tars" / "memory" / "domains.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"domains": {"../escape": {"keywords": ["x"]}}}))
    with pytest.raises(ValueError):
        load_domain_map(tmp_path)


# ── the map as a value ────────────────────────────────────────────────────


def test_a_map_can_be_built_from_scratch_for_a_test_or_a_project() -> None:
    tiny = DomainMap(
        domains=(
            Domain(name="widget", description="widgets", keywords=("widget",)),
            Domain(name="gadget", description="gadgets", paths=("src/gadget",)),
        )
    )
    assert tiny.domains_for("the widget in src/gadget/x.py") == ("gadget", "widget")


def test_a_map_refuses_two_domains_with_the_same_name() -> None:
    with pytest.raises(ValueError, match="widget"):
        DomainMap(
            domains=(
                Domain(name="widget", description="a", keywords=("a",)),
                Domain(name="widget", description="b", keywords=("b",)),
            )
        )


def test_a_map_serialises_back_to_the_override_format(tmp_path) -> None:
    # Round-tripping means a project can dump the default map, edit it, and
    # keep it — the only realistic way anyone tunes this.
    data = DEFAULT_DOMAINS.to_dict()
    path = tmp_path / ".tars" / "memory" / "domains.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    assert load_domain_map(tmp_path).names == DEFAULT_DOMAINS.names
