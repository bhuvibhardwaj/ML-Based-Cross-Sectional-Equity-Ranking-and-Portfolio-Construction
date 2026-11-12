"""
Headless QA harness for the 7-page Streamlit app (`app.py`).

Uses streamlit.testing.v1.AppTest to execute each navigation page in a fresh
script run and reports any uncaught exception, plus element counts so empty
pages are detectable. Also exercises the interactive selectors on Pages 3, 5, 7.

Run:  .venv/bin/python tests/qa_app_pages.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")

PAGES = [
    "1. Project Overview",
    "2. Dataset Explorer",
    "3. Feature Analysis",
    "4. Model Training & Comparison",
    "5. Cross-Sectional Stock Ranking",
    "6. Ranking Performance",
    "7. Portfolio Evaluation",
]


def _run(at: AppTest) -> AppTest:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at.run()
    dep = [str(w.message) for w in caught if "deprecat" in str(w.message).lower()]
    for msg in sorted(set(dep)):
        print(f"    DEPRECATION: {msg[:160]}")
    return at


def _report(label: str, at: AppTest) -> bool:
    n_df = len(at.dataframe)
    n_charts = len(at.get("plotly_chart"))
    n_metric = len(at.metric)
    ok = len(at.exception) == 0
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: dataframes={n_df} plotly_charts={n_charts} metrics={n_metric}")
    for exc in at.exception:
        print("    EXCEPTION:", exc.message)
        for line in exc.stack_trace[-6:]:
            print("     ", line.rstrip())
    return ok


def main() -> int:
    all_ok = True
    for page in PAGES:
        at = AppTest.from_file(APP, default_timeout=180)
        _run(at)
        if at.exception:
            all_ok &= _report(f"initial load before {page}", at)
            continue
        at.sidebar.radio[0].set_value(page)
        _run(at)
        all_ok &= _report(page, at)

        # Exercise interactive selectors on each page
        if page.startswith("3."):
            for feat in at.selectbox[0].options[:13]:
                at.selectbox[0].set_value(feat)
                _run(at)
                if at.exception:
                    all_ok &= _report(f"{page} [feature={feat}]", at)
            print(f"    exercised {len(at.selectbox[0].options)} feature selections")
        if page.startswith("5."):
            for split in at.selectbox[0].options:
                at.selectbox[0].set_value(split)
                _run(at)
                months = at.selectbox[1].options
                for model in at.selectbox[2].options:
                    for month in (months[0], months[-1]):
                        at.selectbox[1].set_value(month)
                        at.selectbox[2].set_value(model)
                        _run(at)
                        if at.exception:
                            all_ok &= _report(f"{page} [{split} | {month} | {model}]", at)
            print("    exercised split x model x first/last month combinations")
        if page.startswith("7."):
            for strat in at.selectbox[0].options:
                at.selectbox[0].set_value(strat)
                _run(at)
                if at.exception:
                    all_ok &= _report(f"{page} [strategy={strat}]", at)
            print(f"    exercised {len(at.selectbox[0].options)} strategy holdings logs")

    print("\nRESULT:", "ALL PAGES PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
