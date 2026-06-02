"""
Hybrid bond search.

  natural_language_search(text):  parses queries like
        "AA bonds maturing within 3 years"
        "bonds above 11% YTM"
        "highest yielding PSU bonds"
    into structured SQL filters + falls back to semantic search.

  filter_search(**filters):       direct structured filtering.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from database.sqlite_store import fetch_all
from database.vector_store import semantic_search

RATING_PATTERN = re.compile(
    r"\b(AAA|AA\+|AA-|AA|A\+|A-|A|BBB\+|BBB-|BBB|BB\+|BB-|BB|B|D)\b",
    re.IGNORECASE,
)
YTM_PATTERN  = re.compile(r"(above|over|>=|>|under|below|<=|<)\s*(\d+\.?\d*)\s*%?\s*(ytm|yield)?", re.I)
MAT_PATTERN  = re.compile(r"(within|under|over|after)\s*(\d+)\s*(year|yr|month)s?", re.I)
SECTOR_HINTS = {
    "psu":      ["psu", "public sector"],
    "banking":  ["bank", "banking", "financial"],
    "infra":    ["infra", "infrastructure", "power", "energy"],
    "auto":     ["auto", "vehicle"],
    "it":       ["it", "tech", "technology", "software"],
}
SUPERLATIVE = re.compile(r"\b(highest|lowest|top|bottom|best|worst)\b", re.I)


def _parse_query(text: str) -> dict:
    f: dict = {}
    if (m := RATING_PATTERN.search(text)):
        f["rating"] = m.group(1).upper()
    if (m := YTM_PATTERN.search(text)):
        op, val = m.group(1).lower(), float(m.group(2))
        if op in ("above", "over", ">", ">="):
            f["min_yield"] = val
        else:
            f["max_yield"] = val
    if (m := MAT_PATTERN.search(text)):
        word, n, unit = m.group(1).lower(), int(m.group(2)), m.group(3).lower()
        days = n * (30 if unit.startswith("month") else 365)
        target = date.today() + timedelta(days=days)
        if word in ("within", "under"):
            f["max_maturity"] = target.isoformat()
        else:
            f["min_maturity"] = target.isoformat()
    for sector, hints in SECTOR_HINTS.items():
        if any(h in text.lower() for h in hints):
            f["sector_like"] = sector
            break
    if (m := SUPERLATIVE.search(text)):
        f["order"] = m.group(1).lower()
    return f


def filter_search(rating: str | None = None,
                  min_yield: float | None = None,
                  max_yield: float | None = None,
                  min_maturity: str | None = None,
                  max_maturity: str | None = None,
                  sector_like: str | None = None,
                  order: str | None = None,
                  limit: int = 50) -> pd.DataFrame:
    df = fetch_all()
    if df.empty:
        return df

    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"], errors="coerce")
    df["Yield"]        = pd.to_numeric(df["Yield"], errors="coerce")
    df["CouponRate"]   = pd.to_numeric(df["CouponRate"], errors="coerce")

    if rating:
        df = df[df["Rating"].astype(str).str.upper().str.startswith(rating.upper())]
    if min_yield is not None:
        df = df[df["Yield"].fillna(df["CouponRate"]) >= min_yield]
    if max_yield is not None:
        df = df[df["Yield"].fillna(df["CouponRate"]) <= max_yield]
    if min_maturity:
        df = df[df["MaturityDate"] >= pd.to_datetime(min_maturity)]
    if max_maturity:
        df = df[df["MaturityDate"] <= pd.to_datetime(max_maturity)]
    if sector_like:
        m = (df["Sector"].astype(str).str.lower().str.contains(sector_like.lower())
             | df["IssuerName"].astype(str).str.lower().str.contains(sector_like.lower())
             | df["BondType"].astype(str).str.lower().str.contains(sector_like.lower()))
        df = df[m]

    if order in ("highest", "top", "best"):
        df = df.sort_values("Yield", ascending=False, na_position="last")
    elif order in ("lowest", "bottom", "worst"):
        df = df.sort_values("Yield", ascending=True, na_position="last")

    return df.head(limit).reset_index(drop=True)


def natural_language_search(text: str, k: int = 25) -> dict:
    filters = _parse_query(text)
    structured = filter_search(**filters, limit=k)

    semantic_hits = []
    try:
        semantic_hits = semantic_search(text, k=k)
    except Exception as e:
        semantic_hits = [{"error": str(e)}]

    return {
        "query":           text,
        "parsed_filters":  filters,
        "structured_rows": structured.to_dict(orient="records"),
        "semantic_hits":   semantic_hits,
        "total":           int(len(structured)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_parse_query("AA bonds maturing within 3 years above 7% YTM"),
                     indent=2))
