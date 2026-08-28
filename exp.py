#!/usr/bin/env python3
"""Safe Python-like expression eval against a dict/object context.

``ExpressionRunner.eval(context, expr)`` supports attribute/item access,
calls (``count(item.rows)`` and ``item.rows.count()``), ``and``/``or``/``not``,
comparisons, ``in``/``not in``, lists, parentheses, ``??`` coalesce, and
``cond ? a : b``. Templates use ``{{ ... }}`` with optional ``:format`` specs.

Context keys may contain spaces (``repeat_for: "Account Status"``,
``{{Account Status}}``). Those are path lookups, not Python identifiers.
"""

from __future__ import annotations

import ast
import html
import logging
import operator
import re
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import date, datetime
from typing import Any, Callable

logger = logging.getLogger("html_table")

_strict_mode: ContextVar[bool] = ContextVar("html_table_strict", default=False)


class HtmlTableError(ValueError):
    """Raised in strict mode for config/data problems that are otherwise warnings."""


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()

_TOKEN_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AGGREGATES = frozenset({"sum", "avg", "min", "max", "count"})
_NAME_ALIASES = {"true": True, "false": False, "null": None, "none": None, "None": None}
_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def in_strict_mode() -> bool:
    return _strict_mode.get()


def set_strict(value: bool):
    return _strict_mode.set(value)


def reset_strict(token: Any) -> None:
    _strict_mode.reset(token)


def fail_or_warn(message: str, *args: Any) -> None:
    if in_strict_mode():
        raise HtmlTableError(message % args if args else message)
    logger.warning(message, *args)


def _truthy(value: Any) -> bool:
    return value is not MISSING and bool(value)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and not isinstance(value, (str, bytes))


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
        if ch in "([{":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch in ")]}":
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
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
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
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
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
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
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


def _parse_call(expr: str) -> tuple[str, str] | None:
    expr = expr.strip()
    open_paren = expr.find("(")
    if open_paren <= 0 or not expr.endswith(")"):
        return None
    name = expr[:open_paren].strip()
    if not _IDENT_RE.match(name):
        return None
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
                return name, expr[open_paren + 1 : i]
    return None


def resolve_path(path: str, context: Any) -> Any:
    """Walk ``a.b.c`` where any segment may be a dict key with spaces."""

    current: Any = context
    for segment in path.split("."):
        current = _attr(current, segment)
        if current is MISSING:
            return MISSING
    return current


def _mapping_lookup(context: Any, key: str) -> Any:
    if _is_mapping(context) and key in context:
        return context[key]
    return MISSING


def _attr(obj: Any, name: str) -> Any:
    if obj is MISSING or obj is None:
        return MISSING
    if name.startswith("_"):
        fail_or_warn("Access to %r is not allowed", name)
        return MISSING
    if _is_mapping(obj) and name in obj:
        return obj[name]
    if isinstance(obj, (list, tuple)) and name.lstrip("-").isdigit():
        idx = int(name)
        if -len(obj) <= idx < len(obj):
            return obj[idx]
        return MISSING
    if hasattr(obj, name):
        return getattr(obj, name)
    return MISSING


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _looks_like_records(items: list | tuple) -> bool:
    for element in items:
        if element is None:
            continue
        if isinstance(element, dict):
            return True
        if isinstance(element, (str, bytes, int, float, bool, list, tuple)):
            return False
        return True
    return False


def _to_number(value: Any) -> float | None:
    if value is MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value)
    return None


def _field_values(items: Any, field: str | None) -> list[Any]:
    if not _is_seq(items):
        return []
    if field is None:
        return list(items)
    out: list[Any] = []
    for element in items:
        out.append(resolve_path(field, element) if field else MISSING)
    return out


def _aggregate(name: str, items: Any, field: str | None) -> Any:
    if name == "count":
        if field is None:
            if items is MISSING or items is None:
                return 0
            if _is_seq(items):
                return len(items)
            try:
                return len(items)
            except TypeError:
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


def _fn_aggregate(name: str):
    def _call(items: Any = None, field: Any = None, *rest: Any) -> Any:
        if rest:
            fail_or_warn("Too many arguments for %s()", name)
        field_name: str | None = None
        if field is not None and field is not MISSING:
            field_name = str(field)
        return _aggregate(name, items, field_name)

    _call.__name__ = name
    return _call


def _fn_coalesce(*values: Any) -> Any:
    for value in values:
        if value is not MISSING and value is not None:
            return value
    return MISSING


def _fn_len(value: Any) -> int:
    if value is MISSING or value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


_BUILTINS: dict[str, Any] = {
    "sum": _fn_aggregate("sum"),
    "avg": _fn_aggregate("avg"),
    "min": _fn_aggregate("min"),
    "max": _fn_aggregate("max"),
    "count": _fn_aggregate("count"),
    "len": _fn_len,
    "abs": abs,
    "round": round,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "sorted": sorted,
    "any": any,
    "all": all,
    "coalesce": _fn_coalesce,
}


def _lookup_name(name: str, context: Any) -> Any:
    if name in _NAME_ALIASES:
        return _NAME_ALIASES[name]
    found = _mapping_lookup(context, name)
    if found is not MISSING:
        return found
    if name in _BUILTINS:
        return _BUILTINS[name]
    if hasattr(context, name) and not name.startswith("_"):
        return getattr(context, name)
    return MISSING


def _call_method(obj: Any, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    if name.startswith("_"):
        fail_or_warn("Call to %r is not allowed", name)
        return MISSING
    if obj is MISSING or obj is None:
        if name in ("sum", "count"):
            return 0
        return MISSING

    if name in _AGGREGATES and _is_seq(obj):
        if name == "count" and args and not _looks_like_records(obj):
            try:
                return list(obj).count(*args, **kwargs)
            except TypeError:
                fail_or_warn("count() failed on sequence")
                return MISSING
        field = None
        if args:
            field = None if args[0] is MISSING or args[0] is None else str(args[0])
        elif "field" in kwargs:
            field = str(kwargs["field"])
        return _aggregate(name, obj, field)

    meth = _attr(obj, name)
    if meth is MISSING or not callable(meth):
        fail_or_warn("Unknown method %r", name)
        return MISSING
    try:
        return meth(*args, **kwargs)
    except TypeError as exc:
        fail_or_warn("Call to %r failed: %s", name, exc)
        return MISSING


def _coerce_comparable(value: Any) -> Any:
    if isinstance(value, (datetime, date, int, float, bool)):
        return value
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value)
    return value


def _norm(value: Any) -> Any:
    return None if value is MISSING else value


class _SafeEval(ast.NodeVisitor):
    def __init__(self, context: Any) -> None:
        self.context = context

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        return _lookup_name(node.id, self.context)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        return _attr(self.visit(node.value), node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        obj = self.visit(node.value)
        key = self.visit(node.slice)
        if obj is MISSING or obj is None or key is MISSING:
            return MISSING
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError):
            return MISSING

    def visit_Slice(self, node: ast.Slice) -> slice:
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not _truthy(operand)
        if operand is MISSING or operand is None:
            return MISSING
        try:
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
        except TypeError:
            return MISSING
        fail_or_warn("Unsupported unary operator")
        return MISSING

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left, right = self.visit(node.left), self.visit(node.right)
        if left is MISSING or right is MISSING:
            return MISSING
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            fail_or_warn("Unsupported operator %s", type(node.op).__name__)
            return MISSING
        try:
            return fn(left, right)
        except (TypeError, ZeroDivisionError, ValueError):
            return MISSING

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            value: Any = True
            for child in node.values:
                value = self.visit(child)
                if not _truthy(value):
                    return False if value is MISSING else value
            return value
        value = False
        for child in node.values:
            value = self.visit(child)
            if _truthy(value):
                return value
        return False if value is MISSING else value

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if not self._compare(op, left, right):
                return False
            left = right
        return True

    def _compare(self, op: ast.cmpop, lhs: Any, rhs: Any) -> bool:
        if isinstance(op, ast.In):
            if rhs is MISSING or rhs is None:
                return False
            try:
                return lhs in rhs
            except TypeError:
                return False
        if isinstance(op, ast.NotIn):
            if rhs is MISSING or rhs is None:
                return True
            try:
                return lhs not in rhs
            except TypeError:
                return True
        left, right = _norm(lhs), _norm(rhs)
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        left_c, right_c = _coerce_comparable(left), _coerce_comparable(right)
        try:
            if isinstance(op, ast.Lt):
                return left_c < right_c
            if isinstance(op, ast.LtE):
                return left_c <= right_c
            if isinstance(op, ast.Gt):
                return left_c > right_c
            if isinstance(op, ast.GtE):
                return left_c >= right_c
        except TypeError:
            return False
        return False

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        branch = node.body if _truthy(self.visit(node.test)) else node.orelse
        return self.visit(branch)

    def visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Set(self, node: ast.Set) -> set[Any]:
        return {self.visit(elt) for elt in node.elts}

    def visit_Dict(self, node: ast.Dict) -> dict[Any, Any]:
        out: dict[Any, Any] = {}
        for key_node, val_node in zip(node.keys, node.values):
            if key_node is None:
                continue
            out[self.visit(key_node)] = self.visit(val_node)
        return out

    def visit_Call(self, node: ast.Call) -> Any:
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
        if isinstance(node.func, ast.Attribute):
            obj = self.visit(node.func.value)
            return _call_method(obj, node.func.attr, args, kwargs)
        func = self.visit(node.func)
        if func is MISSING or not callable(func):
            fail_or_warn("Unknown function in expression")
            return MISSING
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            fail_or_warn("Function call failed: %s", exc)
            return MISSING

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                inner = self.visit(value.value)
                spec = ""
                if value.format_spec is not None:
                    spec = self.visit(value.format_spec)
                    spec = spec if isinstance(spec, str) else str(spec)
                parts.append(_format_value(inner, spec))
            else:
                visited = self.visit(value)
                parts.append("" if visited is MISSING or visited is None else str(visited))
        return "".join(parts)

    def generic_visit(self, node: ast.AST) -> Any:
        fail_or_warn("Unsupported expression node %s", type(node).__name__)
        return MISSING


def _eval_legacy_call(context: Any, expr: str) -> Any:
    parsed = _parse_call(expr)
    if not parsed:
        return MISSING
    name, args_src = parsed
    func = _BUILTINS.get(name)
    if func is None:
        fail_or_warn("Unknown function %r in expression", name)
        return MISSING
    args = [eval_expr(context, part.strip()) for part in _split_top_level(args_src, ",") if part.strip()]
    try:
        return func(*args)
    except TypeError as exc:
        fail_or_warn("Function call failed: %s", exc)
        return MISSING


def _eval_python(context: Any, expr: str) -> Any:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        found = _mapping_lookup(context, expr)
        if found is not MISSING:
            return found
        resolved = resolve_path(expr, context)
        if resolved is not MISSING:
            return resolved
        called = _eval_legacy_call(context, expr)
        if called is not MISSING:
            return called
        fail_or_warn("Invalid expression %r", expr)
        return MISSING
    return _SafeEval(context).visit(tree.body)


def eval_expr(context: Any, expr: str) -> Any:
    """Evaluate ``expr`` against ``context`` (dict, object, or nested mix)."""

    expr = expr.strip()
    if not expr:
        return MISSING
    match = _TOKEN_RE.fullmatch(expr)
    if match:
        expr = match.group(1).strip()
        if not expr:
            return MISSING

    unwrapped = _unwrap_parens(expr)
    if unwrapped != expr:
        return eval_expr(context, unwrapped)

    question = _find_ternary_question(expr)
    if question != -1:
        condition = expr[:question]
        remainder = expr[question + 1 :]
        colon = _find_ternary_colon(remainder)
        when_true, when_false = (remainder, "") if colon == -1 else (remainder[:colon], remainder[colon + 1 :])
        branch = when_true if eval_condition(condition.strip(), context) else when_false
        return eval_expr(context, branch.strip())

    parts = _split_top_level(expr, "??")
    if len(parts) > 1:
        for part in parts:
            value = eval_expr(context, part.strip())
            if value is not MISSING and value is not None:
                return value
        return MISSING

    found = _mapping_lookup(context, expr)
    if found is not MISSING:
        return found

    return _eval_python(context, expr)


def eval_path(path: str, context: Any) -> Any:
    return eval_expr(context, path)


def eval_condition(when: str | None, context: Any) -> bool:
    if not when:
        return False
    return _truthy(eval_expr(context, when))


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
        remainder = content[question + 1 :]
        colon = _find_ternary_colon(remainder)
        when_true, when_false = (remainder, "") if colon == -1 else (remainder[:colon], remainder[colon + 1 :])
        branch = when_true if eval_condition(condition.strip(), context) else when_false
        return _eval_token(branch.strip(), context)

    colon = _find_format_colon(content)
    if colon == -1:
        expr, spec = content, ""
    else:
        expr, spec = content[:colon], content[colon + 1 :]
    value = eval_expr(context, expr.strip())
    return value, _format_value(value, spec.strip())


def render_template(template: str | None, context: Any, *, escape: bool = True) -> str:
    if not template:
        return ""
    if "{{" not in template:
        return html.escape(template, quote=True) if escape else template

    def _sub(match: re.Match[str]) -> str:
        _, text = _eval_token(match.group(1).strip(), context)
        return html.escape(text, quote=True) if escape else text

    return _TOKEN_RE.sub(_sub, template)


class ExpressionRunner:
    """Evaluate expressions and templates against a context mapping/object."""

    def eval(self, context: Any, expr: str) -> Any:
        return eval_expr(context, expr)

    def eval_condition(self, context: Any, expr: str | None) -> bool:
        return eval_condition(expr, context)

    def render(self, context: Any, template: str | None, *, escape: bool = True) -> str:
        return render_template(template, context, escape=escape)


__all__ = [
    "MISSING",
    "HtmlTableError",
    "ExpressionRunner",
    "eval_expr",
    "eval_path",
    "eval_condition",
    "render_template",
    "resolve_path",
    "fail_or_warn",
    "set_strict",
    "reset_strict",
    "in_strict_mode",
]
