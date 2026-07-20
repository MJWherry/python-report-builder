#!/usr/bin/env python3
"""
HTML table builder driven by a single Pydantic/JSON config and a small
``{{ ... }}`` template language.

Pipeline::

    HtmlTableConfig (JSON)  +  data (dict OR object)
                |
                v
    HtmlGenerator.build(config, data)
        Phase 1: _resolve  -> RenderedTable (+ title / base_css)
        Phase 2: _render   -> HTML string (one <table id="…">)

Supports named/inline styles, ``style_rules``, column defaults (visual/
colspan-aware), ``repeat_for`` with ``filter_when`` / multi-key ``sort_by`` /
``limit``, ``index``/``index1``, ``css_class``, unique ``table_id`` for
scoping ``base_css`` in multi-table email HTML, and optional ``strict`` mode.

See README.md for the full config schema and template language.

Requires Pydantic v2::

    pip install pydantic
"""

from __future__ import annotations

import html
import logging
import re
import secrets
from contextvars import ContextVar
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Active during HtmlGenerator.build when config.strict (or build(strict=...)) is on.
_strict_mode: ContextVar[bool] = ContextVar("html_table_strict", default=False)


class HtmlTableError(ValueError):
    """Raised in strict mode for config/data problems that are otherwise warnings."""


def _in_strict_mode() -> bool:
    return _strict_mode.get()


def _fail_or_warn(message: str, *args: Any) -> None:
    """Log a warning, or raise :class:`HtmlTableError` when strict mode is on."""

    if _in_strict_mode():
        raise HtmlTableError(message % args if args else message)
    logger.warning(message, *args)


# ---------------------------------------------------------------------------
# Style enums (closed CSS keyword sets; lengths/colors stay free strings)
# ---------------------------------------------------------------------------

class TextAlign(str, Enum):
    left = "left"
    right = "right"
    center = "center"
    justify = "justify"
    start = "start"
    end = "end"


class VerticalAlign(str, Enum):
    top = "top"
    middle = "middle"
    bottom = "bottom"
    baseline = "baseline"
    text_top = "text-top"
    text_bottom = "text-bottom"
    sub = "sub"
    superscript = "super"  # JSON/CSS value remains "super"


class FontWeight(str, Enum):
    normal = "normal"
    bold = "bold"
    bolder = "bolder"
    lighter = "lighter"
    w100 = "100"
    w200 = "200"
    w300 = "300"
    w400 = "400"
    w500 = "500"
    w600 = "600"
    w700 = "700"
    w800 = "800"
    w900 = "900"


class FontStyle(str, Enum):
    normal = "normal"
    italic = "italic"
    oblique = "oblique"


class BorderStyle(str, Enum):
    none = "none"
    hidden = "hidden"
    dotted = "dotted"
    dashed = "dashed"
    solid = "solid"
    double = "double"
    groove = "groove"
    ridge = "ridge"
    inset = "inset"
    outset = "outset"


class WhiteSpace(str, Enum):
    normal = "normal"
    nowrap = "nowrap"
    pre = "pre"
    pre_wrap = "pre-wrap"
    pre_line = "pre-line"
    break_spaces = "break-spaces"


class TextDecoration(str, Enum):
    """Single-token decorations. Compound values (e.g. ``underline solid``) use ``extra_css``."""

    none = "none"
    underline = "underline"
    overline = "overline"
    line_through = "line-through"


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class Style(BaseModel):
    """Cell/row/table styling. Each field maps to one inline CSS declaration."""

    extends: str | None = None  # named-style composition only (ignored as CSS)

    border: str | None = None
    border_top: str | None = None
    border_right: str | None = None
    border_bottom: str | None = None
    border_left: str | None = None
    border_color: str | None = None
    border_width: str | None = None
    border_style: BorderStyle | None = None

    background_color: str | None = None
    color: str | None = None
    text_align: TextAlign | None = None
    vertical_align: VerticalAlign | None = None
    text_decoration: TextDecoration | None = None
    white_space: WhiteSpace | None = None
    line_height: str | None = None

    font_weight: FontWeight | None = None
    font_size: str | None = None
    font_family: str | None = None
    font_style: FontStyle | None = None

    padding: str | None = None
    padding_top: str | None = None
    padding_right: str | None = None
    padding_bottom: str | None = None
    padding_left: str | None = None

    margin: str | None = None
    margin_top: str | None = None
    margin_right: str | None = None
    margin_bottom: str | None = None
    margin_left: str | None = None

    width: str | None = None
    min_width: str | None = None
    max_width: str | None = None
    height: str | None = None
    opacity: str | None = None

    extra_css: dict[str, str] = Field(default_factory=dict)


class StyleRule(BaseModel):
    """Conditional style applied when ``when`` evaluates truthy."""

    when: str
    style: Style | None = None
    style_name: str | None = None

    @model_validator(mode="after")
    def _require_style_or_name(self) -> StyleRule:
        if self.style is None and not self.style_name:
            raise ValueError("StyleRule requires style or style_name")
        return self


class ColumnConfig(BaseModel):
    """Defaults applied to cells by visual column index (colspan-aware)."""

    style: Style | None = None
    style_name: str | None = None
    empty_text: str | None = None
    value: str | None = None  # default cell value template when cell.value is ""
    css_class: str | None = None


class CellConfig(BaseModel):
    value: str = ""
    link: str | None = None
    style: Style | None = None
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
    style: Style | None = None
    style_name: str | None = None
    style_rules: list[StyleRule] = Field(default_factory=list)
    hide_when: str | None = None
    filter_when: str | None = None  # drop items from repeat_for when truthy
    repeat_for: str | None = None
    item_alias: str = "item"
    sort_by: str | list[str] | None = None
    sort_desc: bool = False  # only when sort_by is a single string
    limit: int | None = None
    css_class: str | None = None


class HtmlTableConfig(BaseModel):
    """Single-table HTML config (one JSON object)."""

    model_config = {"extra": "forbid"}

    id: str | None = None  # <table id>; auto ht_xxxxxxxx if omitted
    title: str | None = None
    caption: str | None = None
    base_css: str | None = None  # templated (use #{{table_id}} … for multi-table)
    styles: dict[str, Style] = Field(default_factory=dict)
    columns: list[ColumnConfig] = Field(default_factory=list)
    headers: list[RowConfig] = Field(default_factory=list)
    rows: list[RowConfig] = Field(default_factory=list)
    footers: list[RowConfig] = Field(default_factory=list)
    table_style: Style | None = None
    default_cell_style: Style | None = None
    default_cell_style_name: str | None = None
    css_class: str | None = None
    striped: bool = False
    stripe_style_name: str | None = None
    strict: bool = False  # raise on unknown styles / bad repeat_for / unknown funcs


# Backward-compatible alias
ReportConfig = HtmlTableConfig


# ---------------------------------------------------------------------------
# Rendered models
# ---------------------------------------------------------------------------

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
    title: str | None = None  # document <h1>
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


# ---------------------------------------------------------------------------
# Template engine
# ---------------------------------------------------------------------------

class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

_TOKEN_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMPARATORS = ("==", "!=", "<=", ">=", "<", ">")
_STRING_OPS = ("not in", "starts_with", "ends_with", "contains", "in")
_AGGREGATES = frozenset({"sum", "avg", "min", "max", "count"})

_DEFAULT_STRIPE = Style(background_color="#f8fafc")


def _resolve_path(path: str, context: Any) -> Any:
    current: Any = context
    for segment in path.split("."):
        if current is MISSING or current is None:
            return MISSING
        if isinstance(current, dict):
            if segment in current:
                current = current[segment]
                continue
            return MISSING
        if isinstance(current, (list, tuple)) and segment.lstrip("-").isdigit():
            idx = int(segment)
            if -len(current) <= idx < len(current):
                current = current[idx]
                continue
            return MISSING
        if hasattr(current, segment):
            current = getattr(current, segment)
            continue
        return MISSING
    return current


def _parse_literal(token: str) -> Any:
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
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    depth = 0
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
        if ch in "([":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and text.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_")
    after_ok = end >= len(text) or not (text[end].isalnum() or text[end] == "_")
    return before_ok and after_ok


def _find_top_level_keyword(text: str, keyword: str) -> int:
    """Leftmost top-level keyword match with word boundaries, or -1."""

    quote: str | None = None
    depth = 0
    i = 0
    n = len(keyword)
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and text.startswith(keyword, i) and _is_word_boundary(text, i, i + n):
            return i
        i += 1
    return -1


def _find_top_level_operator(text: str, op: str) -> int:
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and text.startswith(op, i):
            if op in ("<", ">") and i + 1 < len(text) and text[i + 1] == "=":
                i += 1
                continue
            if op == "=" and i + 1 < len(text) and text[i + 1] == "=":
                i += 1
                continue
            return i
        i += 1
    return -1


def _find_format_colon(text: str) -> int:
    quote: str | None = None
    depth = 0
    ternary_depth = 0
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            if ch == "?":
                if i + 1 < len(text) and text[i + 1] == "?":
                    i += 2
                    continue
                ternary_depth += 1
                i += 1
                continue
            if ch == ":":
                if ternary_depth > 0:
                    ternary_depth -= 1
                    i += 1
                    continue
                return i
        i += 1
    return -1


def _find_ternary_question(text: str) -> int:
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and ch == "?":
            if i + 1 < len(text) and text[i + 1] == "?":
                i += 2
                continue
            return i
        i += 1
    return -1


def _find_ternary_colon(text: str) -> int:
    quote: str | None = None
    depth = 0
    ternary_depth = 0
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            if ch == "?":
                if i + 1 < len(text) and text[i + 1] == "?":
                    i += 2
                    continue
                ternary_depth += 1
                i += 1
                continue
            if ch == ":":
                if ternary_depth == 0:
                    return i
                ternary_depth -= 1
        i += 1
    return -1


def _unwrap_parens(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        quote: str | None = None
        wraps = True
        for i, ch in enumerate(expr):
            if quote:
                if ch == quote:
                    quote = None
                continue
            if ch in "\"'":
                quote = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    wraps = False
                    break
        if not wraps or depth != 0:
            break
        expr = expr[1:-1].strip()
    return expr


def _parse_list_literal(expr: str, context: Any) -> Any | None:
    expr = expr.strip()
    if not (expr.startswith("[") and expr.endswith("]")):
        return None
    inner = expr[1:-1].strip()
    if not inner:
        return []
    items = []
    for part in _split_top_level(inner, ","):
        part = part.strip()
        if not part:
            continue
        items.append(_eval_expr_raw(part, context))
    return items


def _parse_call(expr: str) -> tuple[str, str] | None:
    expr = expr.strip()
    open_paren = expr.find("(")
    if open_paren <= 0 or not expr.endswith(")"):
        return None
    name = expr[:open_paren].strip()
    if not _IDENT_RE.match(name):
        return None
    # ensure the closing ) matches the opening (
    depth = 0
    quote: str | None = None
    for i in range(open_paren, len(expr)):
        ch = expr[i]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if i != len(expr) - 1:
                    return None
                return name, expr[open_paren + 1:i]
    return None


def _field_values(items: Any, field: str | None) -> list[Any]:
    if items is MISSING or items is None:
        return []
    if not isinstance(items, (list, tuple)):
        return []
    if field is None:
        return list(items)
    out = []
    for element in items:
        val = _resolve_path(field, element)
        out.append(val)
    return out


def _to_number(value: Any) -> float | None:
    if value is MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value)
    return None


def _eval_aggregate(name: str, args_src: str, context: Any) -> Any:
    parts = [p.strip() for p in _split_top_level(args_src, ",") if p.strip()]
    if not parts:
        return MISSING
    items = _eval_expr_raw(parts[0], context)
    field: str | None = None
    if len(parts) >= 2:
        field_val = _eval_expr_raw(parts[1], context)
        if field_val is MISSING or field_val is None:
            return MISSING
        field = str(field_val)

    if name == "count":
        if field is None:
            if items is MISSING or items is None:
                return 0
            if isinstance(items, (list, tuple)):
                return len(items)
            return 0
        values = _field_values(items, field)
        return sum(1 for v in values if v is not MISSING and v is not None)

    values = _field_values(items, field)
    nums = [n for n in (_to_number(v) for v in values) if n is not None]
    if name == "sum":
        return sum(nums) if nums else 0
    if not nums:
        return MISSING
    if name == "avg":
        return sum(nums) / len(nums)
    if name == "min":
        return min(nums)
    if name == "max":
        return max(nums)
    return MISSING


def _eval_primary(expr: str, context: Any) -> Any:
    expr = expr.strip()
    if not expr:
        return MISSING

    list_val = _parse_list_literal(expr, context)
    if list_val is not None:
        return list_val

    call = _parse_call(expr)
    if call:
        name, args_src = call
        if name in _AGGREGATES:
            return _eval_aggregate(name, args_src, context)
        _fail_or_warn("Unknown function %r in expression", name)
        return MISSING

    literal = _parse_literal(expr)
    if literal is not MISSING or expr.strip().lower() in ("null", "none"):
        return literal
    return _resolve_path(expr.strip(), context)


def _eval_coalesce(expr: str, context: Any) -> Any:
    for operand in _split_top_level(expr, "??"):
        value = _eval_primary(operand.strip(), context)
        if value is not MISSING and value is not None:
            return value
    return MISSING


def _as_str(value: Any) -> str | None:
    if value is MISSING or value is None:
        return None
    return str(value)


def _compare(lhs: Any, op: str, rhs: Any) -> bool:
    if op == "in":
        if rhs is MISSING or rhs is None:
            return False
        try:
            return lhs in rhs
        except TypeError:
            return False
    if op == "not in":
        if rhs is MISSING or rhs is None:
            return True
        try:
            return lhs not in rhs
        except TypeError:
            return True
    if op == "contains":
        left, right = _as_str(lhs), _as_str(rhs)
        return bool(left is not None and right is not None and right in left)
    if op == "starts_with":
        left, right = _as_str(lhs), _as_str(rhs)
        return bool(left is not None and right is not None and left.startswith(right))
    if op == "ends_with":
        left, right = _as_str(lhs), _as_str(rhs)
        return bool(left is not None and right is not None and left.endswith(right))

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


def _coerce_comparable(value: Any) -> Any:
    if isinstance(value, (datetime, date, int, float, bool)):
        return value
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value)
    return value


def _eval_comparison(expr: str, context: Any) -> Any:
    for op in _STRING_OPS:
        idx = _find_top_level_keyword(expr, op)
        if idx != -1:
            lhs = _eval_coalesce(expr[:idx].strip(), context)
            rhs = _eval_coalesce(expr[idx + len(op):].strip(), context)
            return _compare(lhs, op, rhs)
    for op in _COMPARATORS:
        idx = _find_top_level_operator(expr, op)
        if idx != -1:
            lhs = _eval_coalesce(expr[:idx].strip(), context)
            rhs = _eval_coalesce(expr[idx + len(op):].strip(), context)
            return _compare(lhs, op, rhs)
    return _eval_coalesce(expr, context)


def _eval_unary_not(expr: str, context: Any) -> Any:
    if len(expr) >= 3 and expr.startswith("not") and _is_word_boundary(expr, 0, 3):
        value = _eval_expr_raw(expr[3:].strip(), context)
        return not (bool(value) and value is not MISSING)
    return _eval_comparison(expr, context)


def _eval_and(expr: str, context: Any) -> Any:
    parts = _split_top_level_keyword(expr, "and")
    if len(parts) == 1:
        return _eval_unary_not(parts[0].strip(), context)
    for part in parts:
        # Re-enter so parenthesized sub-expressions keep full precedence.
        value = _eval_expr_raw(part.strip(), context)
        if not (bool(value) and value is not MISSING):
            return False
    return True


def _eval_or(expr: str, context: Any) -> Any:
    parts = _split_top_level_keyword(expr, "or")
    if len(parts) == 1:
        return _eval_and(parts[0].strip(), context)
    for part in parts:
        value = _eval_expr_raw(part.strip(), context)
        if bool(value) and value is not MISSING:
            return True
    return False


def _split_top_level_keyword(text: str, keyword: str) -> list[str]:
    parts: list[str] = []
    buf_start = 0
    quote: str | None = None
    depth = 0
    i = 0
    n = len(keyword)
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
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and text.startswith(keyword, i) and _is_word_boundary(text, i, i + n):
            parts.append(text[buf_start:i])
            i += n
            buf_start = i
            continue
        i += 1
    parts.append(text[buf_start:])
    return parts


def _eval_expr_raw(expr: str, context: Any) -> Any:
    expr = expr.strip()
    if not expr:
        return MISSING
    unwrapped = _unwrap_parens(expr)
    if unwrapped != expr:
        # Re-enter full precedence so and/or bind inside former parentheses.
        return _eval_expr_raw(unwrapped, context)
    return _eval_or(expr, context)


def _format_value(value: Any, spec: str) -> str:
    if value is MISSING or value is None:
        return ""
    if not spec:
        return str(value)
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return str(value)


def _eval_token(content: str, context: Any) -> tuple[Any, str]:
    content = content.strip()

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
    if not template:
        return ""

    def _sub(match: re.Match[str]) -> str:
        _, text = _eval_token(match.group(1).strip(), context)
        return html.escape(text, quote=True) if escape else text

    return _TOKEN_RE.sub(_sub, template)


def eval_path(path: str, context: Any) -> Any:
    path = path.strip()
    inner = path
    match = _TOKEN_RE.fullmatch(path)
    if match:
        inner = match.group(1).strip()
    return _eval_expr_raw(inner, context)


def eval_condition(when: str | None, context: Any) -> bool:
    if not when:
        return False
    value = eval_path(when, context)
    return bool(value) and value is not MISSING


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

_STYLE_FIELDS: dict[str, str] = {
    "border": "border",
    "border_top": "border-top",
    "border_right": "border-right",
    "border_bottom": "border-bottom",
    "border_left": "border-left",
    "border_color": "border-color",
    "border_width": "border-width",
    "border_style": "border-style",
    "background_color": "background-color",
    "color": "color",
    "text_align": "text-align",
    "vertical_align": "vertical-align",
    "text_decoration": "text-decoration",
    "white_space": "white-space",
    "line_height": "line-height",
    "font_weight": "font-weight",
    "font_size": "font-size",
    "font_family": "font-family",
    "font_style": "font-style",
    "padding": "padding",
    "padding_top": "padding-top",
    "padding_right": "padding-right",
    "padding_bottom": "padding-bottom",
    "padding_left": "padding-left",
    "margin": "margin",
    "margin_top": "margin-top",
    "margin_right": "margin-right",
    "margin_bottom": "margin-bottom",
    "margin_left": "margin-left",
    "width": "width",
    "min_width": "min-width",
    "max_width": "max-width",
    "height": "height",
    "opacity": "opacity",
}


def _style_to_dict(style: Style | None) -> dict[str, str]:
    if style is None:
        return {}
    out: dict[str, str] = {}
    for field_name, css_name in _STYLE_FIELDS.items():
        value = getattr(style, field_name)
        if value is not None:
            # str Enum members are already str; normalize anything else
            out[css_name] = value.value if isinstance(value, Enum) else str(value)
    out.update(style.extra_css)
    return out


def _css_to_str(css: dict[str, str]) -> str:
    return ";".join(f"{k}:{v}" for k, v in css.items())


def _resolve_named_style(
    name: str | None,
    registry: dict[str, Style],
    *,
    stack: list[str] | None = None,
) -> dict[str, str]:
    if not name:
        return {}
    stack = stack or []
    if name in stack:
        chain = " -> ".join([*stack, name])
        _fail_or_warn("Style extends cycle detected: %s", chain)
        return {}
    style = registry.get(name)
    if style is None:
        _fail_or_warn("Unknown style_name %r", name)
        return {}
    merged: dict[str, str] = {}
    if style.extends:
        merged.update(_resolve_named_style(style.extends, registry, stack=[*stack, name]))
    merged.update(_style_to_dict(style))
    return merged


def _apply_style_ref(
    css: dict[str, str],
    *,
    style_name: str | None,
    style: Style | None,
    registry: dict[str, Style],
) -> None:
    if style_name:
        css.update(_resolve_named_style(style_name, registry))
    if style is not None:
        # inline style may itself declare extends — resolve against registry
        if style.extends:
            css.update(_resolve_named_style(style.extends, registry))
        css.update(_style_to_dict(style))


class _Desc:
    """Wrap a value so ascending sort yields descending order for that component."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Desc):
            return NotImplemented
        try:
            return self.value > other.value  # inverted
        except TypeError:
            return str(self.value) > str(other.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Desc):
            return NotImplemented
        return self.value == other.value


# ---------------------------------------------------------------------------
# HtmlGenerator
# ---------------------------------------------------------------------------

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
        """Build one HTML table fragment.

        ``strict`` overrides ``config.strict`` when passed. In strict mode,
        unknown ``style_name``, ``extends`` cycles, bad ``repeat_for``, and
        unknown template functions raise :class:`HtmlTableError`.
        """

        config = self._coerce_config(config)
        is_strict = config.strict if strict is None else strict
        token = _strict_mode.set(is_strict)
        try:
            rendered = self._resolve(config, data)
            html_out = self._render(rendered)
        finally:
            _strict_mode.reset(token)
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
        """Alias for :meth:`build` (single-table API)."""

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
            # Ensure table_id is not overwritten by data keys
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
        style: Style | None,
        style_name: str | None,
        config: HtmlTableConfig,
    ) -> dict[str, str]:
        css: dict[str, str] = {}
        _apply_style_ref(
            css,
            style_name=style_name,
            style=style,
            registry=config.styles,
        )
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
        # Remaining rowspan occupancy per visual column (within this section).
        blocked: list[int] = []

        for row in rows:
            if row.repeat_for:
                items = eval_path(row.repeat_for, context)
                if items is MISSING or items is None:
                    _fail_or_warn(
                        "repeat_for path %r resolved to nothing; emitting no rows",
                        row.repeat_for,
                    )
                    items = []
                if not isinstance(items, (list, tuple)):
                    _fail_or_warn(
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

                self.logger.debug(
                    "Expanding %r into %d row(s)", row.repeat_for, len(items)
                )
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
                val = _resolve_path(path, element)
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
        # Copy occupancy; place cells in free columns, then decrement for next row.
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
            resolved = self._resolve_cell(
                cell, config, context, tag=tag, col_index=cursor
            )
            if resolved is not None:
                cells.append(resolved)
            for i in range(span):
                occ[cursor + i] = max(occ[cursor + i], cell.rowspan)
            cursor += span

        # End of row: one less remaining rowspan in every column.
        next_blocked = [max(0, n - 1) for n in occ]

        # All cells hidden → drop the row (avoids empty <tr>).
        if not cells and row.cells:
            return None, next_blocked

        css: dict[str, str] = {}
        if stripe:
            stripe_name = config.stripe_style_name
            if stripe_name:
                css.update(_resolve_named_style(stripe_name, config.styles))
            else:
                css.update(_style_to_dict(_DEFAULT_STRIPE))

        _apply_style_ref(
            css,
            style_name=row.style_name,
            style=row.style,
            registry=config.styles,
        )
        for rule in row.style_rules:
            if eval_condition(rule.when, context):
                _apply_style_ref(
                    css,
                    style_name=rule.style_name,
                    style=rule.style,
                    registry=config.styles,
                )

        return (
            RenderedRow(
                cells=cells,
                style_css=_css_to_str(css),
                css_class=row.css_class,
            ),
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
        if column:
            _apply_style_ref(
                css,
                style_name=column.style_name,
                style=column.style,
                registry=config.styles,
            )
        _apply_style_ref(
            css,
            style_name=config.default_cell_style_name,
            style=config.default_cell_style,
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


__all__ = [
    "TextAlign",
    "VerticalAlign",
    "FontWeight",
    "FontStyle",
    "BorderStyle",
    "WhiteSpace",
    "TextDecoration",
    "Style",
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
    "render_template",
    "eval_path",
    "eval_condition",
]
