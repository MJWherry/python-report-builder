#!/usr/bin/env python3
"""
Declarative data extraction driven by Pydantic config.

``DataExtractor`` composes two runners:

- ``RequestsRunner`` - all API I/O via the ``requests`` library.
- ``PlaywrightRunner`` - browser scraping and the one-time manual login that
  captures ``storage_state``.

Sources (``api`` / ``playwright``) declare a ``base_url``; each ``ExtractionConfig``
references a source, returns a scalar or a custom Pydantic model (mapping fields
to JSON paths / selectors), and optionally carries an ``AuthProbe``. Auth is
captured once via a headed Playwright login, persisted as ``storage_state``, and
reused by ``requests`` (cookies + optional bearer header) and by Playwright pages
until auth is required again.

Requires::

    pip install pydantic requests playwright
    playwright install chromium   # only needed for scraping / manual login
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Union

import requests
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    api = "api"
    playwright = "playwright"


class Source(BaseModel):
    id: str
    type: SourceType
    base_url: str


class ScalarType(str, Enum):
    str_ = "str"
    int_ = "int"
    float_ = "float"
    bool_ = "bool"
    decimal_ = "decimal"
    datetime_ = "datetime"


class ReturnSpec(BaseModel):
    kind: Literal["scalar", "custom"] = "scalar"
    scalar_type: ScalarType = ScalarType.str_      # used when kind == scalar
    model: str | None = None                       # custom: registry key or dotted import path
    many: bool = False                             # return a list of scalars/models


class FieldMapping(BaseModel):
    name: str                                      # target model field
    locator: str                                   # api: JSON path; playwright: CSS/XPath selector
    scalar_type: ScalarType = ScalarType.str_
    remove_chars: list[str] = Field(default_factory=list)
    datetime_format: str | None = None


class AuthProbe(BaseModel):
    login_url: str | None = None                   # default: source.base_url
    storage_state_path: str | None = None          # default: <auth_dir>/<source_id>.json
    # "logged in" signal (manual login finishes when matched / timed out):
    success_url_pattern: str | None = None         # regex on page.url
    success_selector: str | None = None
    # "auth is required again" signal:
    auth_required_selector: str | None = None      # e.g. visible SSO/login button
    auth_required_url_pattern: str | None = None
    unauthorized_status: list[int] = Field(default_factory=lambda: [401, 403])  # api probe
    login_timeout_seconds: int = 300
    # optional bearer auth for APIs that put a token in localStorage instead of cookies:
    token_local_storage_key: str | None = None     # localStorage key in storage_state origins
    token_header: str = "Authorization"            # header to inject for requests
    token_prefix: str = "Bearer "                  # value prefix


class _ExtractionBase(BaseModel):
    id: str
    source_id: str
    name: str                                      # readable name
    returns: ReturnSpec = Field(default_factory=ReturnSpec)
    auth: AuthProbe | None = None                  # None => public / no auth
    fields: list[FieldMapping] = Field(default_factory=list)  # required when returns.kind == custom
    container: str | None = None                   # many=True: JSON path / row selector for items


class ApiExtraction(_ExtractionBase):
    type: Literal["api"] = "api"
    route: str                                     # appended to source.base_url
    method: str = "GET"
    query: dict[str, str] = Field(default_factory=dict)
    value_path: str | None = None                  # scalar: JSON path to the single value
    remove_chars: list[str] = Field(default_factory=list)


class PlaywrightExtraction(_ExtractionBase):
    type: Literal["playwright"] = "playwright"
    selectors: list[str] = Field(default_factory=list)  # scalar / list-of-scalars
    remove_chars: list[str] = Field(default_factory=list)
    wait_for: str | None = None                    # selector to await before extracting


ExtractionConfig = Annotated[
    Union[ApiExtraction, PlaywrightExtraction],
    Field(discriminator="type"),
]


class ExtractorConfig(BaseModel):
    sources: list[Source] = Field(default_factory=list)
    extractions: list[ExtractionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_refs(self) -> "ExtractorConfig":
        types = {s.id: s.type for s in self.sources}
        for e in self.extractions:
            if e.source_id not in types:
                raise ValueError(f"extraction {e.id!r}: unknown source_id {e.source_id!r}")
            if types[e.source_id].value != e.type:
                raise ValueError(
                    f"extraction {e.id!r}: type {e.type!r} != source type {types[e.source_id].value!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ModelResolver = Callable[[str], "type[BaseModel]"]


def _json_path(obj: Any, path: str | None) -> Any:
    """Resolve a dotted JSON path with dict keys and list indices."""

    if not path:
        return obj
    current = obj
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            if segment in current:
                current = current[segment]
                continue
            return None
        if isinstance(current, (list, tuple)) and segment.lstrip("-").isdigit():
            idx = int(segment)
            if -len(current) <= idx < len(current):
                current = current[idx]
                continue
            return None
        return None
    return current


def _trim(value: Any, remove_chars: list[str]) -> Any:
    """Remove configured substrings and trim whitespace (strings only)."""

    if not isinstance(value, str):
        return value
    for token in remove_chars:
        value = value.replace(token, "")
    return value.strip()


def _cast(value: Any, scalar_type: ScalarType, datetime_format: str | None = None) -> Any:
    if value is None:
        return None
    text = value.strip() if isinstance(value, str) else value
    try:
        if scalar_type is ScalarType.str_:
            return str(text)
        if scalar_type is ScalarType.int_:
            return int(float(text)) if isinstance(text, str) else int(text)
        if scalar_type is ScalarType.float_:
            return float(text)
        if scalar_type is ScalarType.decimal_:
            return Decimal(str(text))
        if scalar_type is ScalarType.bool_:
            if isinstance(text, bool):
                return text
            return str(text).strip().lower() in ("true", "1", "yes", "y", "on")
        if scalar_type is ScalarType.datetime_:
            if isinstance(text, datetime):
                return text
            if datetime_format:
                return datetime.strptime(str(text), datetime_format)
            return datetime.fromisoformat(str(text))
    except (ValueError, TypeError, ArithmeticError) as exc:
        logger.warning("Cast of %r to %s failed: %s", value, scalar_type.value, exc)
        return None
    return value


def _build_model(
    record: Any,
    fields: list[FieldMapping],
    model: "type[BaseModel]",
    getter: Callable[[Any, str], Any],
) -> BaseModel:
    data: dict[str, Any] = {}
    for field in fields:
        raw = _trim(getter(record, field.locator), field.remove_chars)
        data[field.name] = _cast(raw, field.scalar_type, field.datetime_format)
    return model(**data)


# ---------------------------------------------------------------------------
# Requests runner (API)
# ---------------------------------------------------------------------------

class RequestsRunner:
    """All API I/O via the requests library, reusing captured storage_state."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def session_for(self, auth: AuthProbe | None) -> requests.Session:
        session = requests.Session()
        if auth and auth.storage_state_path and Path(auth.storage_state_path).exists():
            state = json.loads(Path(auth.storage_state_path).read_text(encoding="utf-8"))
            for cookie in state.get("cookies", []):
                session.cookies.set(
                    cookie["name"],
                    cookie["value"],
                    domain=cookie.get("domain"),
                    path=cookie.get("path", "/"),
                )
            if auth.token_local_storage_key:
                token = self._token_from_state(state, auth.token_local_storage_key)
                if token is not None:
                    session.headers[auth.token_header] = f"{auth.token_prefix}{token}"
                else:
                    self.logger.warning(
                        "localStorage key %r not found in storage_state", auth.token_local_storage_key
                    )
        return session

    @staticmethod
    def _token_from_state(state: dict[str, Any], key: str) -> str | None:
        for origin in state.get("origins", []):
            for entry in origin.get("localStorage", []):
                if entry.get("name") == key:
                    return entry.get("value")
        return None

    def _url(self, source: Source, route: str) -> str:
        return f"{source.base_url.rstrip('/')}/{route.lstrip('/')}"

    def probe(self, ext: ApiExtraction, source: Source, session: requests.Session) -> bool:
        if ext.auth is None:
            return True
        resp = session.request(
            ext.method, self._url(source, ext.route), params=ext.query, allow_redirects=True
        )
        authorized = resp.status_code not in ext.auth.unauthorized_status
        self.logger.debug("Probe %s -> %s (authorized=%s)", ext.id, resp.status_code, authorized)
        return authorized

    def run(
        self,
        ext: ApiExtraction,
        source: Source,
        session: requests.Session,
        resolve_model: ModelResolver,
    ) -> Any:
        url = self._url(source, ext.route)
        self.logger.debug("GET %s params=%s", url, ext.query)
        resp = session.request(ext.method, url, params=ext.query)
        resp.raise_for_status()
        data = resp.json()
        return self._map(ext, data, resolve_model)

    def _map(self, ext: ApiExtraction, data: Any, resolve_model: ModelResolver) -> Any:
        spec = ext.returns

        if spec.many:
            items = _json_path(data, ext.container) or []
            if not isinstance(items, (list, tuple)):
                self.logger.warning("container %r did not resolve to a list", ext.container)
                items = []
            if spec.kind == "custom":
                model = resolve_model(spec.model or "")
                return [_build_model(item, ext.fields, model, _json_path) for item in items]
            return [
                _cast(_trim(_json_path(item, ext.value_path), ext.remove_chars), spec.scalar_type)
                for item in items
            ]

        if spec.kind == "custom":
            model = resolve_model(spec.model or "")
            record = _json_path(data, ext.container) if ext.container else data
            return _build_model(record, ext.fields, model, _json_path)

        raw = _trim(_json_path(data, ext.value_path), ext.remove_chars)
        return _cast(raw, spec.scalar_type)


# ---------------------------------------------------------------------------
# Playwright runner (scraping + manual login)
# ---------------------------------------------------------------------------

class PlaywrightRunner:
    """Browser scraping plus the manual-login step that writes storage_state."""

    def __init__(self, logger: logging.Logger, headless: bool = True) -> None:
        self.logger = logger
        self.headless = headless
        self._pw = None
        self._browser = None

    def __enter__(self) -> "PlaywrightRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_started(self) -> None:
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright  # lazy import

        self.logger.debug("Launching Chromium (headless=%s)", self.headless)
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
            self._pw = None

    @staticmethod
    def _state_arg(auth: AuthProbe | None) -> str | None:
        if auth and auth.storage_state_path and Path(auth.storage_state_path).exists():
            return auth.storage_state_path
        return None

    def needs_auth(self, ext: PlaywrightExtraction, source: Source) -> bool:
        if ext.auth is None:
            return False
        self._ensure_started()
        context = self._browser.new_context(storage_state=self._state_arg(ext.auth))
        try:
            page = context.new_page()
            page.goto(source.base_url)
            auth = ext.auth
            if auth.auth_required_url_pattern and re.search(auth.auth_required_url_pattern, page.url):
                return True
            if auth.auth_required_selector:
                return page.locator(auth.auth_required_selector).count() > 0
            return False
        finally:
            context.close()

    def manual_login(self, auth: AuthProbe, source: Source) -> None:
        self._ensure_started()
        login_url = auth.login_url or source.base_url
        browser = self._pw.chromium.launch(headless=False)  # login is always headed
        try:
            context = browser.new_context(storage_state=self._state_arg(auth))
            page = context.new_page()
            self.logger.info("Opening %s for manual login", login_url)
            page.goto(login_url)
            self._wait_for_login(page, auth)
            path = Path(auth.storage_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(path))
            self.logger.info("Saved storage_state -> %s", path)
        finally:
            browser.close()

    def _wait_for_login(self, page: Any, auth: AuthProbe) -> None:
        timeout_ms = auth.login_timeout_seconds * 1000
        if auth.success_url_pattern:
            page.wait_for_url(re.compile(auth.success_url_pattern), timeout=timeout_ms)
        elif auth.success_selector:
            page.wait_for_selector(auth.success_selector, timeout=timeout_ms)
        else:
            input("Complete the login in the browser, then press Enter here to continue...")

    def run(
        self,
        ext: PlaywrightExtraction,
        source: Source,
        resolve_model: ModelResolver,
    ) -> Any:
        self._ensure_started()
        context = self._browser.new_context(storage_state=self._state_arg(ext.auth))
        try:
            page = context.new_page()
            self.logger.debug("goto %s", source.base_url)
            page.goto(source.base_url)
            if ext.wait_for:
                page.wait_for_selector(ext.wait_for)
            return self._map(ext, page, resolve_model)
        finally:
            context.close()

    def _text(self, scope: Any, selector: str, remove_chars: list[str]) -> Any:
        locator = scope.locator(selector)
        if locator.count() == 0:
            self.logger.warning("selector %r matched nothing", selector)
            return None
        return _trim(locator.first.inner_text(), remove_chars)

    def _map(self, ext: PlaywrightExtraction, page: Any, resolve_model: ModelResolver) -> Any:
        spec = ext.returns

        if spec.many:
            if not ext.container:
                self.logger.warning("many=True requires a container row selector")
                return []
            rows = page.locator(ext.container)
            count = rows.count()
            results: list[Any] = []
            for i in range(count):
                row = rows.nth(i)
                if spec.kind == "custom":
                    model = resolve_model(spec.model or "")
                    getter = lambda scope, sel: self._text(scope, sel, ext.remove_chars)
                    results.append(_build_model(row, ext.fields, model, getter))
                else:
                    raw = _trim(row.inner_text(), ext.remove_chars)
                    results.append(_cast(raw, spec.scalar_type))
            return results

        if spec.kind == "custom":
            model = resolve_model(spec.model or "")
            getter = lambda scope, sel: self._text(scope, sel, ext.remove_chars)
            return _build_model(page, ext.fields, model, getter)

        selector = ext.selectors[0] if ext.selectors else ""
        raw = self._text(page, selector, ext.remove_chars)
        return _cast(raw, spec.scalar_type)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DataExtractor:
    """Resolves a source, ensures auth, and dispatches to the matching runner."""

    def __init__(
        self,
        config: ExtractorConfig,
        models: dict[str, "type[BaseModel]"] | None = None,
        logger: logging.Logger | None = None,
        auth_dir: str = ".auth",
        headless: bool = True,
    ) -> None:
        self.config = config
        self.models = models or {}
        self.logger = logger or globals()["logger"]
        self.auth_dir = auth_dir
        self.requests = RequestsRunner(self.logger)
        self.playwright = PlaywrightRunner(self.logger, headless)
        self._extractions = {e.id: e for e in config.extractions}
        self._sources = {s.id: s for s in config.sources}
        self._apply_auth_defaults()

    def __enter__(self) -> "DataExtractor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.playwright.close()

    def _apply_auth_defaults(self) -> None:
        for ext in self.config.extractions:
            if ext.auth and not ext.auth.storage_state_path:
                ext.auth.storage_state_path = str(Path(self.auth_dir) / f"{ext.source_id}.json")

    def _source(self, source_id: str) -> Source:
        return self._sources[source_id]

    def _resolve_model(self, name: str) -> "type[BaseModel]":
        if name in self.models:
            return self.models[name]
        if "." in name:
            module_name, _, attr = name.rpartition(".")
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        raise KeyError(
            f"Unknown model {name!r}: pass it via models={{}} or use a dotted import path"
        )

    def extract(self, extraction_id: str) -> Any:
        ext = self._extractions[extraction_id]
        source = self._source(ext.source_id)
        self.logger.debug("Extracting %s (%s) from %s", ext.id, ext.name, source.id)
        self._ensure_auth(ext, source)
        if isinstance(ext, ApiExtraction):
            session = self.requests.session_for(ext.auth)
            return self.requests.run(ext, source, session, self._resolve_model)
        return self.playwright.run(ext, source, self._resolve_model)

    def extract_all(self) -> dict[str, Any]:
        return {eid: self.extract(eid) for eid in self._extractions}

    def _ensure_auth(self, ext: ExtractionConfig, source: Source) -> None:
        auth = ext.auth
        if auth is None:
            return
        path = auth.storage_state_path
        need = not (path and Path(path).exists())
        if not need:
            if isinstance(ext, ApiExtraction):
                session = self.requests.session_for(auth)
                need = not self.requests.probe(ext, source, session)
            else:
                need = self.playwright.needs_auth(ext, source)
        if need:
            self.logger.warning("Auth required for %s; launching manual login", ext.id)
            self.playwright.manual_login(auth, source)


__all__ = [
    "SourceType",
    "Source",
    "ScalarType",
    "ReturnSpec",
    "FieldMapping",
    "AuthProbe",
    "ApiExtraction",
    "PlaywrightExtraction",
    "ExtractionConfig",
    "ExtractorConfig",
    "RequestsRunner",
    "PlaywrightRunner",
    "DataExtractor",
]
