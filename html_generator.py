#!/usr/bin/env python3
"""
HTML report/table builder driven by Pydantic config models and a small
``{{ ... }}`` template language.

Pipeline (two parts + builder)::

    ReportConfig (templated)  +  data (dict OR object)
                |
                v
    ReportBuilder.build(config, data)
        Phase 1: _resolve  -> RenderedReport (in-memory generated data)
        Phase 2: _render   -> HTML string

Template mini-language. Everything (paths, comparisons, formatting, coalescing,
ternary) happens inside ``{{ ... }}`` - used identically in cell ``value`` /
``link`` and style-rule ``when``:

- Dot-notation paths over dicts, objects and list indices: ``{{item.user.name}}``,
  ``{{items.0.id}}``.
- Python format specs after a colon: ``{{item.amount:,.2f}}``, ``{{pct:.1%}}``,
  ``{{item.when:%Y-%m-%d}}``.
- Null-coalescing with ``??``: ``{{item.note ?? item.fallback ?? "-"}}``.
- Comparisons (``== != < <= > >=``) yielding a boolean: ``{{item.balance < 0}}``.
- Ternary ``cond ? a : b`` (right-associative, may nest and combine with the
  above): ``{{item.balance < 0 ? "PAST DUE" : item.note ?? "-"}}``.
- Literal text may surround any number of tokens: ``"Total: {{total:,}} USD"``.

Requires Pydantic v2::

    pip install pydantic
"""

from __future__ import annotations

import html
import logging
import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Part 1a: Config models (Pydantic, templated)
# ---------------------------------------------------------------------------

class Style(BaseModel):
    """Cell/row/table styling. Each field maps to one inline CSS declaration."""

    border: str | None = None              # "1px solid #ccc"
    background_color: str | None = None
    color: str | None = None
    text_align: str | None = None          # left / right / center
    font_weight: str | None = None
    padding: str | None = None
    width: str | None = None
    extra_css: dict[str, str] = Field(default_factory=dict)  # escape hatch


class StyleRule(BaseModel):
    """Conditional style applied when ``when`` evaluates truthy."""

    when: str                              # full in-brace expr, e.g. "{{item.balance < 0}}"
    style: Style


class CellConfig(BaseModel):
    value: str = ""                        # template, e.g. "{{item.amount:,.2f}}"
    link: str | None = None                # template resolved to an href
    style: Style | None = None
    style_rules: list[StyleRule] = Field(default_factory=list)
    is_header: bool = False                # render as <th> instead of <td>
    colspan: int = 1
    rowspan: int = 1
    raw: bool = False                      # skip HTML-escaping of the value


class RowConfig(BaseModel):
    cells: list[CellConfig] = Field(default_factory=list)
    style: Style | None = None
    repeat_for: str | None = None          # path to a list, e.g. "report.items"
    item_alias: str = "item"               # name bound to each element


class TableConfig(BaseModel):
    title: str | None = None
    headers: list[RowConfig] = Field(default_factory=list)   # <thead>
    rows: list[RowConfig] = Field(default_factory=list)      # <tbody>
    footers: list[RowConfig] = Field(default_factory=list)   # <tfoot>
    table_style: Style | None = None
    default_cell_style: Style | None = None
    css_class: str | None = None


class ReportConfig(BaseModel):
    tables: list[TableConfig] = Field(default_factory=list)
    title: str | None = None
    base_css: str | None = None            # emitted inside a <style> block


# ---------------------------------------------------------------------------
# Part 2 (model): In-memory rendered data
# ---------------------------------------------------------------------------

class RenderedCell(BaseModel):
    html: str
    tag: str                               # "td" | "th"
    style_css: str
    colspan: int = 1
    rowspan: int = 1


class RenderedRow(BaseModel):
    cells: list[RenderedCell] = Field(default_factory=list)
    style_css: str = ""


class RenderedTable(BaseModel):
    title: str | None = None
    thead: list[RenderedRow] = Field(default_factory=list)
    tbody: list[RenderedRow] = Field(default_factory=list)
    tfoot: list[RenderedRow] = Field(default_factory=list)
    table_style_css: str = ""
    css_class: str | None = None


class RenderedReport(BaseModel):
    title: str | None = None
    tables: list[RenderedTable] = Field(default_factory=list)
    base_css: str | None = None


# ---------------------------------------------------------------------------
# Part 1b: Template engine
# ---------------------------------------------------------------------------

class _Missing:
    """Sentinel for an unresolved path (distinct from an explicit ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

_TOKEN_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)$")
_COMPARATORS = ("==", "!=", "<=", ">=", "<", ">")


def _resolve_path(path: str, context: Any) -> Any:
    """Walk ``path`` over dicts, sequences (numeric segments) and objects."""

    current: Any = context
    for segment in path.split("."):
        if current is MISSING or current is None:
            return MISSING
        # dict-style access first
        if isinstance(current, dict):
            if segment in current:
                current = current[segment]
                continue
            return MISSING
        # list/tuple index access
        if isinstance(current, (list, tuple)) and segment.lstrip("-").isdigit():
            idx = int(segment)
            if -len(current) <= idx < len(current):
                current = current[idx]
                continue
            return MISSING
        # object attribute access
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        return MISSING
    return current


def _parse_literal(token: str) -> Any:
    """Parse a quoted string / number / bool / null, else return ``MISSING``."""

    token = token.strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if _NUMBER_RE.match(token):
        return float(token) if ("." in token) else int(token)
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    return MISSING


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` ignoring occurrences inside single/double quotes."""

    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if text.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _find_format_colon(text: str) -> int:
    """Index of the first top-level ``:`` (outside quotes), or -1."""

    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            continue
        if ch == ":":
            return i
    return -1


def _find_ternary_question(text: str) -> int:
    """Index of the first top-level ``?`` that is not part of ``??``, or -1."""

    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == "?":
            if i + 1 < len(text) and text[i + 1] == "?":  # coalescing operator
                i += 2
                continue
            return i
        i += 1
    return -1


def _find_ternary_colon(text: str) -> int:
    """Index of the ``:`` matching the current ternary level (handles nesting)."""

    quote: str | None = None
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == "?":
            if i + 1 < len(text) and text[i + 1] == "?":  # coalescing operator
                i += 2
                continue
            depth += 1
            i += 1
            continue
        if ch == ":":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return -1


def _eval_operand(token: str, context: Any) -> Any:
    """Resolve a single operand: literal if it looks like one, else a path."""

    literal = _parse_literal(token)
    if literal is not MISSING or token.strip().lower() in ("null", "none"):
        return literal
    return _resolve_path(token.strip(), context)


def _eval_coalesce(expr: str, context: Any) -> Any:
    """First non-missing/None operand of a ``a ?? b ?? c`` chain."""

    for operand in _split_top_level(expr, "??"):
        value = _eval_operand(operand, context)
        if value is not MISSING and value is not None:
            return value
    return MISSING


def _eval_expr_raw(expr: str, context: Any) -> Any:
    """Evaluate an in-brace expression: comparison (-> bool) or coalescing value.

    No formatting is applied here; ternary and format specs are handled by the
    token layer so everything still resolves inside a single ``{{ ... }}``.
    """

    expr = expr.strip()
    for op in _COMPARATORS:
        idx = _find_top_level_operator(expr, op)
        if idx != -1:
            lhs = _eval_coalesce(expr[:idx].strip(), context)
            rhs = _eval_coalesce(expr[idx + len(op):].strip(), context)
            return _compare(lhs, op, rhs)
    return _eval_coalesce(expr, context)


def _format_value(value: Any, spec: str) -> str:
    if value is MISSING or value is None:
        return ""
    if not spec:
        return str(value)
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        # e.g. numeric spec applied to a string -> degrade gracefully
        return str(value)


def _eval_token(content: str, context: Any) -> tuple[Any, str]:
    """Return ``(value, formatted_text)`` for one ``{{ ... }}`` token body."""

    content = content.strip()

    # Ternary: "cond ? when_true : when_false" (checked before format specs so a
    # branch's own ``:`` format spec is not mistaken for the ternary separator).
    question = _find_ternary_question(content)
    if question != -1:
        condition = content[:question]
        remainder = content[question + 1:]
        colon = _find_ternary_colon(remainder)
        if colon == -1:
            when_true, when_false = remainder, ""
        else:
            when_true, when_false = remainder[:colon], remainder[colon + 1:]
        branch = when_true if eval_condition(condition.strip(), context) else when_false
        return _eval_token(branch.strip(), context)

    colon = _find_format_colon(content)
    if colon == -1:
        expr, spec = content, ""
    else:
        expr, spec = content[:colon], content[colon + 1:]
    value = _eval_expr_raw(expr.strip(), context)
    return value, _format_value(value, spec.strip())


def render_template(template: str | None, context: Any, *, escape: bool = True) -> str:
    """Render every ``{{ ... }}`` token in ``template`` against ``context``."""

    if not template:
        return ""

    def _sub(match: re.Match[str]) -> str:
        _, text = _eval_token(match.group(1).strip(), context)
        return html.escape(text, quote=True) if escape else text

    return _TOKEN_RE.sub(_sub, template)


def eval_path(path: str, context: Any) -> Any:
    """Resolve a bare or ``{{ }}``-wrapped expression to its raw value."""

    path = path.strip()
    inner = path
    match = _TOKEN_RE.fullmatch(path)
    if match:
        inner = match.group(1).strip()
    return _eval_expr_raw(inner, context)


def _coerce_comparable(value: Any) -> Any:
    if isinstance(value, (datetime, date, int, float, bool)):
        return value
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value)
    return value


def eval_condition(when: str | None, context: Any) -> bool:
    """Evaluate a ``{{ ... }}`` expression for truthiness.

    The expression (comparison, coalescing, path) lives entirely inside the
    braces, e.g. ``{{item.balance < 0}}`` or ``{{item.active}}``. A bare
    (brace-less) expression is also accepted for convenience.
    """

    if not when:
        return False
    value = eval_path(when, context)
    return bool(value) and value is not MISSING


def _find_top_level_operator(text: str, op: str) -> int:
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if text.startswith(op, i):
            # avoid matching "<" inside "<=" / ">=" and "=" inside "==" / "!="
            if op in ("<", ">") and i + 1 < len(text) and text[i + 1] == "=":
                i += 1
                continue
            return i
        i += 1
    return -1


def _compare(lhs: Any, op: str, rhs: Any) -> bool:
    if lhs is MISSING:
        lhs = None
    if rhs is MISSING:
        rhs = None
    if op == "==":
        return lhs == rhs
    if op == "!=":
        return lhs != rhs
    left, right = _coerce_comparable(lhs), _coerce_comparable(rhs)
    try:
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Style -> CSS helpers
# ---------------------------------------------------------------------------

_STYLE_FIELDS: dict[str, str] = {
    "border": "border",
    "background_color": "background-color",
    "color": "color",
    "text_align": "text-align",
    "font_weight": "font-weight",
    "padding": "padding",
    "width": "width",
}


def _style_to_dict(style: Style | None) -> dict[str, str]:
    if style is None:
        return {}
    out: dict[str, str] = {}
    for field_name, css_name in _STYLE_FIELDS.items():
        value = getattr(style, field_name)
        if value is not None:
            out[css_name] = value
    out.update(style.extra_css)
    return out


def _merge_styles(*styles: Style | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for style in styles:
        merged.update(_style_to_dict(style))
    return merged


def _css_to_str(css: dict[str, str]) -> str:
    return ";".join(f"{k}:{v}" for k, v in css.items())


# ---------------------------------------------------------------------------
# Part 2 (logic): HtmlGenerator
# ---------------------------------------------------------------------------

class HtmlGenerator:
    """Generates HTML from templated config + data.

    - :meth:`build` renders a whole :class:`ReportConfig` (every table).
    - :meth:`build_table` renders a single :class:`TableConfig` to an HTML table.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or globals()["logger"]

    def build(self, config: ReportConfig, data: dict[str, Any] | Any) -> str:
        self.logger.debug("Building report with %d table(s)", len(config.tables))
        html_out = self._render(self._resolve(config, data))
        self.logger.debug("Report built (%d chars)", len(html_out))
        return html_out

    def build_table(self, table: TableConfig, data: dict[str, Any] | Any) -> str:
        """Generate a single HTML ``<table>`` from one :class:`TableConfig`."""

        self.logger.debug("Building table %r", table.title)
        context = self._build_context(data)
        rendered = self._resolve_table(table, context)
        return self._render_table(rendered)

    # -- Phase 1: templates + data -> in-memory rendered model --------------

    def _resolve(self, config: ReportConfig, data: dict[str, Any] | Any) -> RenderedReport:
        context = self._build_context(data)
        tables = [self._resolve_table(t, context) for t in config.tables]
        return RenderedReport(
            title=render_template(config.title, context),
            tables=tables,
            base_css=config.base_css,
        )

    @staticmethod
    def _build_context(data: dict[str, Any] | Any) -> dict[str, Any]:
        context: dict[str, Any] = {"report": data, "data": data}
        if isinstance(data, dict):
            context.update(data)
        return context

    def _resolve_table(self, table: TableConfig, context: dict[str, Any]) -> RenderedTable:
        return RenderedTable(
            title=render_template(table.title, context),
            thead=self._resolve_section(table.headers, table, context),
            tbody=self._resolve_section(table.rows, table, context),
            tfoot=self._resolve_section(table.footers, table, context),
            table_style_css=_css_to_str(_merge_styles(table.table_style)),
            css_class=table.css_class,
        )

    def _resolve_section(
        self,
        rows: list[RowConfig],
        table: TableConfig,
        context: dict[str, Any],
    ) -> list[RenderedRow]:
        rendered: list[RenderedRow] = []
        for row in rows:
            if row.repeat_for:
                items = eval_path(row.repeat_for, context)
                if items is MISSING or items is None:
                    self.logger.warning(
                        "repeat_for path %r resolved to nothing; emitting no rows",
                        row.repeat_for,
                    )
                    items = []
                self.logger.debug("Expanding %r into %d row(s)", row.repeat_for, len(items))
                for element in items:
                    child = dict(context)
                    child[row.item_alias] = element
                    rendered.append(self._resolve_row(row, table, child))
            else:
                rendered.append(self._resolve_row(row, table, context))
        return rendered

    def _resolve_row(
        self,
        row: RowConfig,
        table: TableConfig,
        context: dict[str, Any],
    ) -> RenderedRow:
        cells = [self._resolve_cell(cell, table, context) for cell in row.cells]
        return RenderedRow(cells=cells, style_css=_css_to_str(_merge_styles(row.style)))

    def _resolve_cell(
        self,
        cell: CellConfig,
        table: TableConfig,
        context: dict[str, Any],
    ) -> RenderedCell:
        text = render_template(cell.value, context, escape=not cell.raw)
        if cell.link:
            href = render_template(cell.link, context, escape=True)
            if href:
                text = f'<a href="{href}">{text}</a>'

        css = _merge_styles(table.default_cell_style, cell.style)
        for rule in cell.style_rules:
            if eval_condition(rule.when, context):
                css.update(_style_to_dict(rule.style))

        return RenderedCell(
            html=text,
            tag="th" if cell.is_header else "td",
            style_css=_css_to_str(css),
            colspan=cell.colspan,
            rowspan=cell.rowspan,
        )

    # -- Phase 2: rendered model -> HTML -----------------------------------

    def _render(self, report: RenderedReport) -> str:
        parts: list[str] = []
        if report.base_css:
            parts.append(f"<style>{report.base_css}</style>")
        if report.title:
            parts.append(f"<h1>{report.title}</h1>")
        for table in report.tables:
            parts.append(self._render_table(table))
        return "\n".join(parts)

    def _render_table(self, table: RenderedTable) -> str:
        attrs = ""
        if table.css_class:
            attrs += f' class="{html.escape(table.css_class, quote=True)}"'
        if table.table_style_css:
            attrs += f' style="{html.escape(table.table_style_css, quote=True)}"'

        out: list[str] = [f"<table{attrs}>"]
        if table.title:
            out.append(f"  <caption>{table.title}</caption>")
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
            style = f' style="{html.escape(row.style_css, quote=True)}"' if row.style_css else ""
            lines.append(f"    <tr{style}>")
            for cell in row.cells:
                lines.append("      " + self._render_cell(cell))
            lines.append("    </tr>")
        return lines

    @staticmethod
    def _render_cell(cell: RenderedCell) -> str:
        attrs = ""
        if cell.style_css:
            attrs += f' style="{html.escape(cell.style_css, quote=True)}"'
        if cell.colspan != 1:
            attrs += f' colspan="{cell.colspan}"'
        if cell.rowspan != 1:
            attrs += f' rowspan="{cell.rowspan}"'
        return f"<{cell.tag}{attrs}>{cell.html}</{cell.tag}>"


__all__ = [
    "Style",
    "StyleRule",
    "CellConfig",
    "RowConfig",
    "TableConfig",
    "ReportConfig",
    "RenderedCell",
    "RenderedRow",
    "RenderedTable",
    "RenderedReport",
    "HtmlGenerator",
    "render_template",
    "eval_path",
    "eval_condition",
]
