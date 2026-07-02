#!/usr/bin/env python3
"""
Run extractions from ``example_extractor_config.json``.

Run::

    python scripts/example_extract.py

By default this runs only the public GitHub API extractions (scalar + custom
model) which need no auth and no browser. The private API and Playwright
scraping extractions are listed too; calling them opens a real browser for a
one-time manual login (storage_state is then reused), so they are guarded behind
``--all``.

Requires::

    pip install pydantic requests playwright
    playwright install chromium   # only for --all (scraping / login)
"""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from data_extractor import DataExtractor, ExtractorConfig

CONFIG_PATH = Path(__file__).resolve().parent / "example_extractor_config.json"


class Repo(BaseModel):
    name: str
    full_name: str
    owner: str
    stars: int


class Account(BaseModel):
    id: int
    name: str
    balance: Decimal


class GrafanaUser(BaseModel):
    id: int
    login: str
    email: str
    name: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="also run the auth/Playwright extractions (opens a browser for login)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    config = ExtractorConfig.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))

    models = {"Repo": Repo, "Account": Account, "GrafanaUser": GrafanaUser}
    with DataExtractor(config, models=models, headless=False) as ex:
        print("flask_stars ->", ex.extract("flask_stars"))
        print("flask_repo  ->", ex.extract("flask_repo"))

        if args.all:
            # Opens Grafana, which redirects to SSO; sign in, then it redirects back
            # and the grafana_session cookie is saved + reused for the API call.
            print("grafana_user      ->", ex.extract("grafana_user"))
            print("example_heading   ->", ex.extract("example_heading"))
            print("dashboard_summary ->", ex.extract("dashboard_summary"))


if __name__ == "__main__":
    main()
