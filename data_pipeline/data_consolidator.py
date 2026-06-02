"""
Data consolidator.

Pulls every available source into a single tabular Bond Master:

  1. Local CSVs in raw_data/        (always used)
       - Combined Bond Indices Yields.csv
       - EMB_data.csv
       - Global finance data.csv
       - prices.csv

  2. Zintellix-collected files in output/processed/   (used if present)
       - nse_debt_master_cleaned.csv
       - live_gsec_bonds_cleaned.csv
       - nselib_debt_securities.csv

Output: output/unified_bond_master.csv  — one row per ISIN (or synthetic key)
with a stable schema downstream code can rely on.
"""

from __future__ import annotations
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR, PROCESSED_DIR, UNIFIED_CSV

UNIFIED_COLUMNS = [
    "ISIN", "Symbol", "IssuerName", "BondType", "Sector", "Country",
    "Currency", "Rating", "CouponRate", "FaceValue", "IssueDate",
    "MaturityDate", "LastTradedPrice", "BidPrice", "AskPrice", "Volume",
    "Yield", "Duration", "Spread", "Source",
]


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[consolidator] could not read {path.name}: {e}")
        return pd.DataFrame()


def _synth_isin(seed: str) -> str:
    """Deterministic synthetic ISIN for rows missing one."""
    h = hashlib.md5(seed.encode()).hexdigest().upper()[:9]
    return f"SYN{h[:9]}"


def _normalise(df: pd.DataFrame, source: str, rename: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    df = df.rename(columns=rename).copy()
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["Source"] = source
    return df[UNIFIED_COLUMNS]


def _load_emb() -> pd.DataFrame:
    df = _safe_read(RAW_DATA_DIR / "EMB_data.csv")
    return _normalise(df, source="EMB_data", rename={"Price": "LastTradedPrice"})


def _load_global_finance() -> pd.DataFrame:
    df = _safe_read(RAW_DATA_DIR / "Global finance data.csv")
    if df.empty:
        return df
    df = df.rename(columns={"Issuer": "IssuerName", "Price": "LastTradedPrice"})
    if "ISIN" not in df.columns:
        df["ISIN"] = df.apply(
            lambda r: _synth_isin(f"{r.get('Symbol','')}-{r.get('Country','')}"),
            axis=1,
        )
    return _normalise(df, source="Global_Finance", rename={})


def _load_prices() -> pd.DataFrame:
    df = _safe_read(RAW_DATA_DIR / "prices.csv")
    if df.empty:
        return df
    df = df.rename(columns={"Ticker": "Symbol", "Close": "LastTradedPrice"})
    return _normalise(df, source="Prices_Feed", rename={})


def _load_bond_indices() -> pd.DataFrame:
    """Indices file has aggregate yields, not individual ISINs.

    We treat each index row as a synthetic 'index bond' so downstream
    analytics can show yield/duration/spread benchmarks alongside real bonds.
    """
    df = _safe_read(RAW_DATA_DIR / "Combined Bond Indices Yields.csv")
    if df.empty:
        return df
    df = df.rename(columns={"Index_Name": "IssuerName"})
    df["Symbol"] = df["IssuerName"].str.replace(" ", "_").str.upper().str[:24]
    df["ISIN"] = df["Symbol"].apply(lambda s: _synth_isin(f"IDX-{s}"))
    df["Rating"] = df["IssuerName"].str.extract(
        r"(AAA|AA\+?|AA|A\+?|BBB|BB|B|D)"
    )
    df["Sector"] = df["BondType"]
    return _normalise(df, source="Bond_Index", rename={})


def _load_zintellix() -> list[pd.DataFrame]:
    out = []
    mapping = {
        "nse_debt_master_cleaned.csv": (
            "NSE_Debt_Master",
            {"FACE VALUE": "FaceValue", "FACEVALUE": "FaceValue",
             "ISSUER NAME": "IssuerName", "COUPON": "CouponRate",
             "MATURITY DATE": "MaturityDate", "ISSUE DATE": "IssueDate",
             "SYMBOL": "Symbol", "SERIES": "Series"},
        ),
        "live_gsec_bonds_cleaned.csv": ("Live_GSec", {}),
        "nselib_debt_securities.csv": ("NSELib_Debt", {}),
    }
    for fname, (src, rename) in mapping.items():
        df = _safe_read(PROCESSED_DIR / fname)
        if not df.empty:
            df.columns = [c.strip() for c in df.columns]
            print(f"[consolidator] including zintellix file {fname} ({len(df)} rows)")
            out.append(_normalise(df, source=src, rename=rename))
    if not out:
        print("[consolidator] no zintellix files found — CSV-only mode")
    return out


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    numeric = ["CouponRate", "FaceValue", "LastTradedPrice", "BidPrice",
               "AskPrice", "Volume", "Yield", "Duration", "Spread"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("IssueDate", "MaturityDate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("ISIN", "Symbol", "IssuerName", "BondType",
              "Sector", "Country", "Currency", "Rating"):
        df[c] = df[c].astype(str).replace({"nan": "", "None": ""}).str.strip()
    return df


def consolidate() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("BOND DATA CONSOLIDATION ENGINE")
    print("=" * 70)

    frames = [
        _load_emb(),
        _load_global_finance(),
        _load_prices(),
        _load_bond_indices(),
        *_load_zintellix(),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No source data found. Check raw_data/ folder.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"[consolidator] raw merged rows: {len(df)}")

    df["ISIN"] = df["ISIN"].astype(str).str.upper().str.strip()
    df.loc[df["ISIN"].isin(["", "NAN", "NONE"]), "ISIN"] = np.nan
    df["ISIN"] = df["ISIN"].fillna(
        df.apply(
            lambda r: _synth_isin(
                f"{r.get('Symbol','')}|{r.get('IssuerName','')}|{r.get('Source','')}"
            ),
            axis=1,
        )
    )

    df = _coerce_types(df)

    df = (
        df.sort_values(["ISIN", "LastTradedPrice"], na_position="last")
          .groupby("ISIN", as_index=False)
          .first()
    )

    UNIFIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIFIED_CSV, index=False)
    print(f"[consolidator] unique ISINs : {len(df)}")
    print(f"[consolidator] sources      : {df['Source'].value_counts().to_dict()}")
    print(f"[consolidator] saved        : {UNIFIED_CSV}")
    print("=" * 70 + "\n")
    return df


def run(try_live: bool = True) -> pd.DataFrame:
    """Consolidate. If try_live, run zintellix collector first (graceful failure)."""
    if try_live:
        try:
            from data_pipeline.zintellix_collector import run_full_collection
            run_full_collection()
        except Exception as e:
            print(f"[consolidator] live ingest skipped: {e}")
    return consolidate()


if __name__ == "__main__":
    run(try_live=True)
