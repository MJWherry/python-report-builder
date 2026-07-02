#!/usr/bin/env python3
"""
Build an HTML table from ``example_report_config.json`` and sample data.

Run::

    python scripts/example_build_report.py

Loads the JSON config into a :class:`ReportConfig` (via Pydantic), then writes
``scripts/example_report.html`` and prints the HTML to stdout. The sample data
deliberately mixes dicts and a plain class instance to show that the template
engine resolves dot-notation across both, plus ``None`` values to exercise
``??`` coalescing and a negative balance to trigger conditional styling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from html_generator import HtmlGenerator, ReportConfig

CONFIG_PATH = Path(__file__).resolve().parent / "example_report_config.json"


@dataclass
class Account:
    """A plain class instance to prove object dot-notation works too."""

    id: int
    name: str
    region: str | None
    amount: float
    balance: float
    note: str | None


def build_sample_data() -> dict[str, object]:
    items = [
        {
            "id": 1,
            "name": "Acme Corp",
            "region": "North",
            "amount": 12450.5,
            "balance": 3200.0,
            "note": "Renewal pending",
        },
        # dict with missing region + None note -> exercises coalescing
        {
            "id": 2,
            "name": "Globex",
            "region": None,
            "amount": 8200.0,
            "balance": -540.25,
            "note": None,
        },
        # object instance -> exercises attribute dot-notation
        Account(
            id=3,
            name="Initech",
            region="West",
            amount=15999.99,
            balance=0.0,
            note="VIP",
        ),
    ]

    total_amount = sum(_amount(i) for i in items)
    total_balance = sum(_balance(i) for i in items)

    return {
        "quarter": "Q2 2026",
        "items": items,
        "totals": {"amount": total_amount, "balance": total_balance},
    }


def _amount(item: object) -> float:
    return item["amount"] if isinstance(item, dict) else item.amount


def _balance(item: object) -> float:
    return item["balance"] if isinstance(item, dict) else item.balance


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    config = ReportConfig.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))
    data = build_sample_data()
    html = HtmlGenerator().build(config, data)

    document = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n<title>Quarterly Sales</title>\n"
        "</head>\n<body>\n" + html + "\n</body>\n</html>\n"
    )

    out_path = Path(__file__).resolve().parent / "example_report.html"
    out_path.write_text(document, encoding="utf-8")

    print(document)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
