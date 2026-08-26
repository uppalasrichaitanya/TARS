"""Efficient, accessibility-tree-first browser testing.

Playwright is deliberately imported only when a :class:`BrowserSession` is
entered.  The serializer is dependency-free so tests and callers can inspect
trees without installing a browser runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class A11yNode:
    node_id: str
    role: str
    name: str = ""
    value: str = ""
    states: tuple[str, ...] = ()
    children: tuple[A11yNode, ...] = ()


class RefRegistry:
    """Session-local deterministic mapping from node identity to small refs."""

    def __init__(self) -> None:
        self._refs: dict[str, int] = {}
        self._next = 1

    def ref_for(self, node_id: str) -> int:
        if node_id not in self._refs:
            self._refs[node_id] = self._next
            self._next += 1
        return self._refs[node_id]


def _clean(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def serialize_tree(
    root: A11yNode,
    refs: RefRegistry | None = None,
    *,
    max_value: int = 120,
) -> str:
    refs = refs or RefRegistry()
    lines: list[str] = []

    def visit(node: A11yNode, depth: int) -> None:
        line = "  " * depth + f"[ref={refs.ref_for(node.node_id)}] {node.role}"
        if node.name:
            line += f' "{_clean(node.name, max_value)}"'
        if node.value:
            line += f' value="{_clean(node.value, max_value)}"'
        if node.states:
            line += " " + " ".join(sorted(set(node.states)))
        lines.append(line)
        for child in node.children:
            visit(child, depth + 1)

    visit(root, 0)
    return "\n".join(lines)


_LINE = re.compile(
    r"^(?P<indent> *)\[ref=(?P<ref>\d+)\] (?P<role>\S+)"
    r'(?: "(?P<name>[^"]*)")?(?P<rest>.*)$'
)


def parse_snapshot(text: str) -> tuple[A11yNode, ...]:
    stack: list[tuple[int, A11yNode]] = []
    roots: list[A11yNode] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        match = _LINE.match(raw)
        if not match:
            raise ValueError(f"invalid snapshot line: {raw!r}")
        rest = match.group("rest").strip()
        value_match = re.search(r'value="([^"]*)"', rest)
        value = value_match.group(1) if value_match else ""
        state_text = re.sub(r'\s*value="[^"]*"', "", rest).strip()
        states = tuple(state_text.split())
        node = A11yNode(
            match.group("ref"),
            match.group("role"),
            match.group("name") or "",
            value,
            states,
        )
        depth = len(match.group("indent")) // 2
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            parent = stack[-1][1]
            object.__setattr__(parent, "children", (*parent.children, node))
        else:
            roots.append(node)
        stack.append((depth, node))
    return tuple(roots)


class BrowserSession:
    def __init__(self, *, headless: bool = True) -> None:
        self.headless = headless
        self._pw: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._cdp: Any = None
        self._refs = RefRegistry()
        self._targets: dict[int, int] = {}

    def __enter__(self) -> Self:
        try:
            from playwright.sync_api import (  # type: ignore[import-not-found]
                sync_playwright,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Playwright is required for browser sessions; install tars[browser]"
            ) from exc
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page()
        self._cdp = self._page.context.new_cdp_session(self._page)
        self._cdp.send("Accessibility.enable")
        self._cdp.send("DOM.enable")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def goto(self, url: str) -> None:
        self._page_or_error().goto(url)

    def _page_or_error(self):
        if self._page is None:
            raise RuntimeError("BrowserSession is not active")
        return self._page

    @staticmethod
    def _ax_value(item: dict[str, Any], key: str) -> Any:
        value = item.get(key, {})
        return value.get("value", "") if isinstance(value, dict) else value

    def _tree(self) -> A11yNode:
        if self._cdp is None:
            raise RuntimeError("BrowserSession is not active")
        raw_nodes = self._cdp.send("Accessibility.getFullAXTree").get("nodes", [])
        included = {
            str(node["nodeId"]): node for node in raw_nodes if not node.get("ignored")
        }
        children: dict[str, list[str]] = {node_id: [] for node_id in included}
        roots: list[str] = []
        for node_id, item in included.items():
            parent = str(item.get("parentId", ""))
            if parent in included:
                children[parent].append(node_id)
            else:
                roots.append(node_id)

        def convert(node_id: str) -> A11yNode:
            item = included[node_id]
            backend = item.get("backendDOMNodeId")
            identity = f"dom:{backend}" if backend is not None else f"ax:{node_id}"
            ref = self._refs.ref_for(identity)
            if backend is not None:
                self._targets[ref] = int(backend)
            properties = {
                str(prop.get("name")): self._ax_value(prop, "value")
                for prop in item.get("properties", [])
            }
            states = []
            for key in ("checked", "expanded", "selected", "pressed"):
                if key in properties:
                    states.append(f"{key}={str(properties[key]).lower()}")
            if properties.get("disabled") is True:
                states.append("disabled")
            return A11yNode(
                identity,
                str(self._ax_value(item, "role") or "generic"),
                str(self._ax_value(item, "name")),
                str(self._ax_value(item, "value")),
                tuple(states),
                tuple(convert(child) for child in children[node_id]),
            )

        if not roots:
            return A11yNode("root", "document")
        converted = tuple(convert(root) for root in roots)
        if len(converted) == 1:
            return converted[0]
        return A11yNode("root", "document", children=converted)

    def snapshot(self) -> str:
        return serialize_tree(self._tree(), self._refs)

    def _locator(self, ref: int):
        backend = self._targets.get(ref)
        if backend is None or self._cdp is None:
            raise KeyError(f"unknown or non-actionable accessibility ref {ref}")
        try:
            resolved = self._cdp.send("DOM.resolveNode", {"backendNodeId": backend})
        except Exception as exc:
            raise KeyError(f"accessibility ref {ref} no longer exists") from exc
        object_id = resolved["object"]["objectId"]
        marker = f"tars-{ref}"
        self._cdp.send(
            "Runtime.callFunctionOn",
            {
                "objectId": object_id,
                "functionDeclaration": "function(v) { this.setAttribute('data-tars-ref', v); }",
                "arguments": [{"value": marker}],
            },
        )
        return self._page_or_error().locator(f'[data-tars-ref="{marker}"]')

    def click(self, ref: int) -> None:
        self._locator(ref).click()

    def type(self, ref: int, text: str) -> None:
        self._locator(ref).fill(text)

    def check(self, ref: int) -> None:
        self._locator(ref).check()

    def uncheck(self, ref: int) -> None:
        self._locator(ref).uncheck()

    def back(self) -> None:
        self._page_or_error().go_back()

    def forward(self) -> None:
        self._page_or_error().go_forward()

    def reload(self) -> None:
        self._page_or_error().reload()

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None


__all__ = [
    "A11yNode",
    "BrowserSession",
    "RefRegistry",
    "parse_snapshot",
    "serialize_tree",
]
