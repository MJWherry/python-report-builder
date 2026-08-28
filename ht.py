#!/usr/bin/env python3
"""
HTML table builder driven by a Pydantic/JSON config and ``{{ ... }}`` templates.

Styles are plain dicts. Keys are normalized to CSS kebab-case
(``border_bottom`` / ``backgroundColor`` → ``border-bottom`` / ``background-color``).
``extends`` is reserved for named-style composition and is not emitted as CSS.

    from html_table import build_table, HtmlGenerator, HtmlTableConfig
"""

from __future__ import annotations

import html
import logging
import re
import secrets
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from expression import (
    MISSING,
    HtmlTableError,
    eval_condition,
    eval_path,
    fail_or_warn,
    render_template,
    reset_strict,
    resolve_path,
    set_strict,
)

logger = logging.getLogger("html_table")

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_DEFAULT_STRIPE = {"background_color": "#f8fafc"}


def normalize_css_key(key: str) -> str:
    """``border_bottom`` / ``borderBottom`` / ``border-bottom`` → ``border-bottom``."""

    key = str(key).strip()
    if key.startswith("--"):
        return key.replace("_", "-")
    return _CAMEL_RE.sub(r"\1-\2", key).replace("_", "-").lower()


def _css_value(value: Any) -> str:
    return str(value.value) if hasattr(value, "value") and not isinstance(value, (str, bytes)) else str(value)


def style_to_css(style: dict[str, Any] | None) -> dict[str, str]:
    if not style:
        return {}
    out: dict[str, str] = {}
    for key, value in style.items():
        if key == "extends" or value is None:
            continue
        if isinstance(value, dict):
            continue
        out[normalize_css_key(key)] = _css_value(value)
    return out


def _css_to_str(css: dict[str, str]) -> str:
    return ";".join(f"{k}:{v}" for k, v in css.items())


def _resolve_named_style(
    name: str | None,
    registry: dict[str, dict[str, Any]],
    *,
    stack: list[str] | None = None,
) -> dict[str, str]:
    if not name:
        return {}
    stack = stack or []
    if name in stack:
        fail_or_warn("Style extends cycle detected: %s", " -> ".join([*stack, name]))
        return {}
    style = registry.get(name)
    if style is None:
        fail_or_warn("Unknown style_name %r", name)
        return {}
    merged: dict[str, str] = {}
    extends = style.get("extends")
    if extends:
        merged.update(_resolve_named_style(str(extends), registry, stack=[*stack, name]))
    merged.update(style_to_css(style))
    return merged


def _apply_style_ref(
    css: dict[str, str],
    *,
    style_name: str | None,
    style: dict[str, Any] | None,
    registry: dict[str, dict[str, Any]],
) -> None:
    if style_name:
        css.update(_resolve_named_style(style_name, registry))
    if style:
        extends = style.get("extends")
        if extends:
            css.update(_resolve_named_style(str(extends), registry))
        css.update(style_to_css(style))


class StyleRule(BaseModel):
    """Conditional style applied when ``when`` evaluates truthy."""

    when: str
    style: dict[str, Any] | None = None
    style_name: str | None = None

    @model_validator(mode="after")
    def _require_style_or_name(self) -> StyleRule:
        if self.style is None and not self.style_name:
            raise ValueError("StyleRule requires style or style_name")
        return self


class ColumnConfig(BaseModel):
    """Defaults applied to cells by visual column index (colspan-aware)."""

    style: dict[str, Any] | None = None
    style_name: str | None = None
    empty_text: str | None = None
    value: str | None = None
    css_class: str | None = None


class CellConfig(BaseModel):
    value: str = ""
    link: str | None = None
    style: dict[str, Any] | None = None
    style_name: str | None = None
    style_rules: list[StyleRule] = Field(default_factory=list)
    empty_text: str | None = None
    hide_when: str | None = None
    css_class: str | None = None
    colspan: int = 1
    rowspan: int = 1
    raw: bool = False


class RowConfig(BaseModel):
    cells: list[CellConfig] = Field(default_factory=list)
    style: dict[str, Any] | None = None
    style_name: str | None = None
    style_rules: list[StyleRule] = Field(default_factory=list)
    hide_when: str | None = None
    filter_when: str | None = None
    repeat_for: str | None = None
    item_alias: str = "item"
    sort_by: str | list[str] | None = None
    sort_desc: bool = False
    limit: int | None = None
    css_class: str | None = None


class HtmlTableConfig(BaseModel):
    """Single-table HTML config (one JSON object)."""

    model_config = {"extra": "forbid"}

    id: str | None = None
    title: str | None = None
    caption: str | None = None
    base_css: str | None = None
    styles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    columns: list[ColumnConfig] = Field(default_factory=list)
    headers: list[RowConfig] = Field(default_factory=list)
    rows: list[RowConfig] = Field(default_factory=list)
    footers: list[RowConfig] = Field(default_factory=list)
    table_style: dict[str, Any] | None = None
    default_cell_style: dict[str, Any] | None = None
    default_cell_style_name: str | None = None
    css_class: str | None = None
    striped: bool = False
    stripe_style_name: str | None = None
    strict: bool = False


ReportConfig = HtmlTableConfig


class RenderedCell(BaseModel):
    html: str
    tag: str
    style_css: str
    css_class: str | None = None
    colspan: int = 1
    rowspan: int = 1


class RenderedRow(BaseModel):
    cells: list[RenderedCell] = Field(default_factory=list)
    style_css: str = ""
    css_class: str | None = None


class RenderedTable(BaseModel):
    title: str | None = None
    caption: str | None = None
    thead: list[RenderedRow] = Field(default_factory=list)
    tbody: list[RenderedRow] = Field(default_factory=list)
    tfoot: list[RenderedRow] = Field(default_factory=list)
    table_style_css: str = ""
    css_class: str | None = None
    table_id: str = ""
    base_css: str | None = None


def _join_css_classes(*parts: str | None) -> str | None:
    seen: list[str] = []
    for part in parts:
        if not part:
            continue
        for token in part.split():
            if token and token not in seen:
                seen.append(token)
    return " ".join(seen) if seen else None


def _new_table_id() -> str:
    return "ht_" + secrets.token_hex(4)


class _Desc:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Desc):
            return NotImplemented
        try:
            return self.value > other.value
        except TypeError:
            return str(self.value) > str(other.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Desc):
            return NotImplemented
        return self.value == other.value


class HtmlGenerator:
    """Generate one HTML table fragment from a single config JSON + data."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or globals()["logger"]

    def build(
        self,
        config: HtmlTableConfig | dict[str, Any] | str,
        data: Any,
        *,
        strict: bool | None = None,
    ) -> str:
        config = self._coerce_config(config)
        is_strict = config.strict if strict is None else strict
        token = set_strict(is_strict)
        try:
            rendered = self._resolve(config, data)
            html_out = self._render(rendered)
        finally:
            reset_strict(token)
        self.logger.debug("Table built (%d chars)", len(html_out))
        return html_out

    def build_from_json(
        self,
        config_json: str | dict[str, Any],
        data: Any,
        *,
        strict: bool | None = None,
    ) -> str:
        return self.build(config_json, data, strict=strict)

    def build_table(
        self,
        config: HtmlTableConfig | dict[str, Any] | str,
        data: Any,
        *,
        strict: bool | None = None,
    ) -> str:
        return self.build(config, data, strict=strict)

    def _coerce_config(self, config: HtmlTableConfig | dict[str, Any] | str) -> HtmlTableConfig:
        if isinstance(config, HtmlTableConfig):
            return config
        if isinstance(config, str):
            return HtmlTableConfig.model_validate_json(config)
        return HtmlTableConfig.model_validate(config)

    @staticmethod
    def _build_context(data: Any, table_id: str) -> dict[str, Any]:
        context: dict[str, Any] = {"report": data, "data": data, "table_id": table_id}
        if isinstance(data, dict):
            context.update(data)
            context["table_id"] = table_id
        return context

    def _resolve(self, config: HtmlTableConfig, data: Any) -> RenderedTable:
        table_id = config.id or _new_table_id()
        context = self._build_context(data, table_id)
        base_css = (
            render_template(config.base_css, context, escape=False)
            if config.base_css
            else None
        )
        return RenderedTable(
            title=render_template(config.title, context),
            caption=render_template(config.caption, context),
            thead=self._resolve_section(config.headers, config, context, section="headers"),
            tbody=self._resolve_section(config.rows, config, context, section="rows"),
            tfoot=self._resolve_section(config.footers, config, context, section="footers"),
            table_style_css=_css_to_str(self._style_dict(config.table_style, None, config)),
            css_class=config.css_class,
            table_id=table_id,
            base_css=base_css,
        )

    def _style_dict(
        self,
        style: dict[str, Any] | None,
        style_name: str | None,
        config: HtmlTableConfig,
    ) -> dict[str, str]:
        css: dict[str, str] = {}
        _apply_style_ref(css, style_name=style_name, style=style, registry=config.styles)
        return css

    def _resolve_section(
        self,
        rows: list[RowConfig],
        config: HtmlTableConfig,
        context: dict[str, Any],
        *,
        section: Literal["headers", "rows", "footers"],
    ) -> list[RenderedRow]:
        tag = "th" if section == "headers" else "td"
        apply_stripe = section == "rows" and config.striped
        rendered: list[RenderedRow] = []
        body_index = 0
        blocked: list[int] = []

        for row in rows:
            if row.repeat_for:
                items = eval_path(row.repeat_for, context)
                if items is MISSING or items is None:
                    fail_or_warn(
                        "repeat_for path %r resolved to nothing; emitting no rows",
                        row.repeat_for,
                    )
                    items = []
                if not isinstance(items, (list, tuple)):
                    fail_or_warn(
                        "repeat_for path %r did not resolve to a list; got %s",
                        row.repeat_for,
                        type(items).__name__,
                    )
                    items = []
                else:
                    items = list(items)

                if row.filter_when:
                    kept: list[Any] = []
                    for element in items:
                        child = dict(context)
                        child[row.item_alias] = element
                        if not eval_condition(row.filter_when, child):
                            kept.append(element)
                    items = kept

                if row.sort_by:
                    items = self._sort_items(items, row.sort_by, row.sort_desc)
                if row.limit is not None:
                    items = items[: max(0, row.limit)]

                self.logger.debug("Expanding %r into %d row(s)", row.repeat_for, len(items))
                for idx, element in enumerate(items):
                    child = dict(context)
                    child[row.item_alias] = element
                    child["index"] = idx
                    child["index1"] = idx + 1
                    if row.hide_when and eval_condition(row.hide_when, child):
                        continue
                    stripe = apply_stripe and (body_index % 2 == 0)
                    resolved, blocked = self._resolve_row(
                        row, config, child, tag=tag, stripe=stripe, blocked=blocked
                    )
                    if resolved is not None:
                        rendered.append(resolved)
                        body_index += 1
            else:
                if row.hide_when and eval_condition(row.hide_when, context):
                    continue
                stripe = apply_stripe and (body_index % 2 == 0)
                resolved, blocked = self._resolve_row(
                    row, config, context, tag=tag, stripe=stripe, blocked=blocked
                )
                if resolved is not None:
                    rendered.append(resolved)
                    if section == "rows":
                        body_index += 1
        return rendered

    def _sort_items(
        self,
        items: list[Any],
        sort_by: str | list[str],
        sort_desc: bool,
    ) -> list[Any]:
        if isinstance(sort_by, str):
            keys: list[tuple[str, bool]] = [(sort_by, sort_desc)]
        else:
            keys = []
            for raw in sort_by:
                desc = raw.startswith("-")
                path = raw[1:] if desc else raw
                if path:
                    keys.append((path, desc))
            if not keys:
                return items

        def multi_key(element: Any, *, as_str: bool) -> tuple[Any, ...]:
            parts: list[Any] = []
            for path, desc in keys:
                val = resolve_path(path, element)
                missing = val is MISSING or val is None
                if missing:
                    parts.append((1, ""))
                    continue
                if as_str:
                    val = str(val)
                if desc:
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        parts.append((0, -val))
                    else:
                        parts.append((0, _Desc(val)))
                else:
                    parts.append((0, val))
            return tuple(parts)

        try:
            return sorted(items, key=lambda e: multi_key(e, as_str=False))
        except TypeError:
            return sorted(items, key=lambda e: multi_key(e, as_str=True))

    def _resolve_row(
        self,
        row: RowConfig,
        config: HtmlTableConfig,
        context: dict[str, Any],
        *,
        tag: str,
        stripe: bool,
        blocked: list[int],
    ) -> tuple[RenderedRow | None, list[int]]:
        cells: list[RenderedCell] = []
        occ = list(blocked)
        cursor = 0

        def _ensure(n: int) -> None:
            while len(occ) < n:
                occ.append(0)

        for cell in row.cells:
            span = max(1, cell.colspan)
            _ensure(cursor + 1)
            while cursor < len(occ) and occ[cursor] > 0:
                cursor += 1
            _ensure(cursor + span)
            resolved = self._resolve_cell(cell, config, context, tag=tag, col_index=cursor)
            if resolved is not None:
                cells.append(resolved)
            for i in range(span):
                occ[cursor + i] = max(occ[cursor + i], cell.rowspan)
            cursor += span

        next_blocked = [max(0, n - 1) for n in occ]
        if not cells and row.cells:
            return None, next_blocked

        css: dict[str, str] = {}
        if stripe:
            if config.stripe_style_name:
                css.update(_resolve_named_style(config.stripe_style_name, config.styles))
            else:
                css.update(style_to_css(_DEFAULT_STRIPE))

        _apply_style_ref(
            css, style_name=row.style_name, style=row.style, registry=config.styles
        )
        for rule in row.style_rules:
            if eval_condition(rule.when, context):
                _apply_style_ref(
                    css, style_name=rule.style_name, style=rule.style, registry=config.styles
                )

        return (
            RenderedRow(cells=cells, style_css=_css_to_str(css), css_class=row.css_class),
            next_blocked,
        )

    def _resolve_cell(
        self,
        cell: CellConfig,
        config: HtmlTableConfig,
        context: dict[str, Any],
        *,
        tag: str,
        col_index: int,
    ) -> RenderedCell | None:
        if cell.hide_when and eval_condition(cell.hide_when, context):
            return None

        column = config.columns[col_index] if col_index < len(config.columns) else None
        value_template = cell.value if cell.value != "" else (
            column.value if column and column.value is not None else ""
        )
        text = render_template(value_template, context, escape=not cell.raw)
        if not text.strip():
            empty = cell.empty_text if cell.empty_text is not None else (
                column.empty_text if column else None
            )
            if empty is not None:
                text = render_template(empty, context, escape=not cell.raw)

        if cell.link:
            href = render_template(cell.link, context, escape=True)
            if href:
                text = f'<a href="{href}">{text}</a>'

        css: dict[str, str] = {}
        _apply_style_ref(
            css,
            style_name=config.default_cell_style_name,
            style=config.default_cell_style,
            registry=config.styles,
        )
        if column:
            _apply_style_ref(
                css,
                style_name=column.style_name,
                style=column.style,
                registry=config.styles,
            )
        _apply_style_ref(
            css,
            style_name=cell.style_name,
            style=cell.style,
            registry=config.styles,
        )
        for rule in cell.style_rules:
            if eval_condition(rule.when, context):
                _apply_style_ref(
                    css,
                    style_name=rule.style_name,
                    style=rule.style,
                    registry=config.styles,
                )

        return RenderedCell(
            html=text,
            tag=tag,
            style_css=_css_to_str(css),
            css_class=_join_css_classes(
                column.css_class if column else None,
                cell.css_class,
            ),
            colspan=cell.colspan,
            rowspan=cell.rowspan,
        )

    def _render(self, table: RenderedTable) -> str:
        parts: list[str] = []
        if table.base_css:
            parts.append(f"<style>{table.base_css}</style>")
        if table.title:
            parts.append(f"<h1>{table.title}</h1>")
        parts.append(self._render_table(table))
        return "\n".join(parts)

    def _render_table(self, table: RenderedTable) -> str:
        attrs = f' id="{html.escape(table.table_id, quote=True)}"'
        if table.css_class:
            attrs += f' class="{html.escape(table.css_class, quote=True)}"'
        if table.table_style_css:
            attrs += f' style="{html.escape(table.table_style_css, quote=True)}"'

        out: list[str] = [f"<table{attrs}>"]
        if table.caption:
            out.append(f"  <caption>{table.caption}</caption>")
        if table.thead:
            out.append("  <thead>")
            out.extend(self._render_rows(table.thead))
            out.append("  </thead>")
        if table.tbody:
            out.append("  <tbody>")
            out.extend(self._render_rows(table.tbody))
            out.append("  </tbody>")
        if table.tfoot:
            out.append("  <tfoot>")
            out.extend(self._render_rows(table.tfoot))
            out.append("  </tfoot>")
        out.append("</table>")
        return "\n".join(out)

    def _render_rows(self, rows: list[RenderedRow]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            attrs = ""
            if row.css_class:
                attrs += f' class="{html.escape(row.css_class, quote=True)}"'
            if row.style_css:
                attrs += f' style="{html.escape(row.style_css, quote=True)}"'
            lines.append(f"    <tr{attrs}>")
            for cell in row.cells:
                lines.append("      " + self._render_cell(cell))
            lines.append("    </tr>")
        return lines

    @staticmethod
    def _render_cell(cell: RenderedCell) -> str:
        attrs = ""
        if cell.css_class:
            attrs += f' class="{html.escape(cell.css_class, quote=True)}"'
        if cell.style_css:
            attrs += f' style="{html.escape(cell.style_css, quote=True)}"'
        if cell.colspan != 1:
            attrs += f' colspan="{cell.colspan}"'
        if cell.rowspan != 1:
            attrs += f' rowspan="{cell.rowspan}"'
        return f"<{cell.tag}{attrs}>{cell.html}</{cell.tag}>"


def build_table(
    config: HtmlTableConfig | dict[str, Any] | str,
    data: Any,
    *,
    strict: bool | None = None,
) -> str:
    """Build one HTML table fragment from config + data."""

    return HtmlGenerator().build(config, data, strict=strict)


__all__ = [
    "StyleRule",
    "ColumnConfig",
    "CellConfig",
    "RowConfig",
    "HtmlTableConfig",
    "ReportConfig",
    "HtmlTableError",
    "RenderedCell",
    "RenderedRow",
    "RenderedTable",
    "HtmlGenerator",
    "build_table",
    "normalize_css_key",
    "style_to_css",
]
