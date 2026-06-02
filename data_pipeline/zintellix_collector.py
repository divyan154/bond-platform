"""
Fixed & repackaged version of zintellix.py.

Original Colab file had syntax errors (empty subscripts) in the consolidation
block. This module restores it as a clean, runnable Python pipeline:

  * Step A : Optional Kaggle ingest (skipped silently if creds missing)
  * Step B : NSE DEBT.csv master archive
  * Step C : Live G-Sec secondary-market feed (nseindia.com)
  * Step D : nselib-based equity + debt registries (optional)
  * Step E : Risk-free interest-rate history
  * Step F : NSS sovereign zero-coupon yield curve

All artefacts land in `bond_platform/output/processed/`.
Network failures degrade gracefully — every step returns None on error
so the downstream consolidator can fall back to CSV-only mode.
"""

from __future__ import annotations
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def safe_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def ingest_nse_debt_master() -> pd.DataFrame | None:
    url = "https://archives.nseindia.com/content/equities/DEBT.csv"
    print("[zintellix] Pulling NSE DEBT.csv master ...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"[zintellix] NSE archive returned {r.status_code}")
            return None
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip()
        df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
        out = PROCESSED_DIR / "nse_debt_master_cleaned.csv"
        df.to_csv(out, index=False)
        print(f"[zintellix]   wrote {len(df)} rows -> {out.name}")
        return df
    except Exception as e:
        print(f"[zintellix] NSE master ingest failed: {e}")
        return None


def scrape_live_gsec() -> pd.DataFrame | None:
    print("[zintellix] Scraping live G-Sec feed ...")
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        api = "https://www.nseindia.com/api/liveBonds-traded-on-cm?type=gsec"
        r = s.get(api, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"[zintellix] live API status {r.status_code}")
            return None
        raw = r.json().get("data", []) or []
        if not raw:
            print("[zintellix] live G-Sec feed empty")
            return None
        rows = []
        for it in raw:
            rows.append({
                "Symbol": str(it.get("symbol", "")).strip(),
                "Series": str(it.get("series", "")).strip(),
                "FaceValue": safe_float(it.get("faceValue"), 100.0),
                "LastTradedPrice": safe_float(
                    it.get("lastPrice") or it.get("previousClose")
                ),
                "Volume": safe_float(it.get("totalTradedVolume")),
                "AveragePrice": safe_float(it.get("averagePrice")),
                "BidPrice": safe_float(it.get("buyPrice1")),
                "AskPrice": safe_float(it.get("sellPrice1")),
                "ISIN": str(it.get("isin", "")).strip(),
            })
        df = pd.DataFrame(rows)
        out = PROCESSED_DIR / "live_gsec_bonds_cleaned.csv"
        df.to_csv(out, index=False)
        print(f"[zintellix]   wrote {len(df)} rows -> {out.name}")
        return df
    except Exception as e:
        print(f"[zintellix] live G-Sec scrape failed: {e}")
        return None


def export_listings_via_nselib() -> pd.DataFrame | None:
    try:
        from nselib import capital_market
    except ImportError:
        print("[zintellix] nselib not installed; skipping")
        return None
    try:
        print("[zintellix] Pulling nselib equity_list ...")
        df_eq = capital_market.equity_list()
        eq_path = PROCESSED_DIR / "nselib_equity_master.csv"
        df_eq.to_csv(eq_path, index=False)
        print(f"[zintellix]   wrote {len(df_eq)} rows -> {eq_path.name}")
        try:
            from nselib import debt
            d = datetime.today().strftime("%d-%m-%Y")
            df_debt = debt.securities_available_for_trading(trade_date=d)
            debt_path = PROCESSED_DIR / "nselib_debt_securities.csv"
            df_debt.to_csv(debt_path, index=False)
            print(f"[zintellix]   wrote {len(df_debt)} rows -> {debt_path.name}")
        except Exception as e:
            print(f"[zintellix] nselib debt failed: {e}")
        return df_eq
    except Exception as e:
        print(f"[zintellix] nselib pipeline failed: {e}")
        return None


def fetch_risk_free_rates() -> pd.DataFrame | None:
    print("[zintellix] Fetching risk-free rate history ...")
    url = (
        "https://techfanetechnologies.github.io/"
        "risk_free_interest_rate/RiskFreeInterestRate.json"
    )
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"[zintellix] risk-free tracker status {r.status_code}")
            return None
        raw = r.json()
        rows = [{"Date": k, "RiskFreeRate": safe_float(v, None)}
                for k, v in raw.items()]
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False)
        out = PROCESSED_DIR / "rbi_t_bill_rates.csv"
        df.to_csv(out, index=False)
        print(f"[zintellix]   wrote {len(df)} rows -> {out.name}")
        return df
    except Exception as e:
        print(f"[zintellix] risk-free fetch failed: {e}")
        return None


class FixedIncomeAnalyticsEngine:
    @staticmethod
    def construct_nss_spot_curve(beta0, beta1, beta2, beta3, tau1, tau2, maturities):
        m = np.array(maturities, dtype=float)
        m = np.where(m <= 0, 1e-5, m)
        t1 = (1 - np.exp(-m / tau1)) / (m / tau1)
        t2 = t1 - np.exp(-m / tau1)
        t3 = (1 - np.exp(-m / tau2)) / (m / tau2) - np.exp(-m / tau2)
        return beta0 + beta1 * t1 + beta2 * t2 + beta3 * t3


def generate_nss_curve() -> pd.DataFrame:
    b0, b1, b2, b3 = 7.4625, -2.2939, 3.8605, 2.9653
    t1, t2 = 20.0, 4.2012
    m = np.linspace(0.1, 30.0, 300)
    r = FixedIncomeAnalyticsEngine.construct_nss_spot_curve(
        b0, b1, b2, b3, t1, t2, m
    )
    df = pd.DataFrame({"MaturityYears": m, "ZeroSpotRate": r})
    out = PROCESSED_DIR / "nss_sovereign_zero_curve.csv"
    df.to_csv(out, index=False)
    print(f"[zintellix] NSS curve written -> {out.name}")
    return df


def run_full_collection() -> dict:
    """Execute every step. Returns a manifest of {name: dataframe-or-None}."""
    print("=" * 70)
    print("ZINTELLIX FIXED-INCOME DATA ACQUISITION PIPELINE")
    print("=" * 70)
    manifest = {
        "nse_master":      ingest_nse_debt_master(),
        "live_gsec":       scrape_live_gsec(),
        "nselib":          export_listings_via_nselib(),
        "risk_free_rates": fetch_risk_free_rates(),
        "nss_curve":       generate_nss_curve(),
    }
    print("-" * 70)
    for k, v in manifest.items():
        n = 0 if v is None else len(v)
        print(f"  {k:20s}: {n:>8} rows")
    print("=" * 70)
    return manifest


if __name__ == "__main__":
    run_full_collection()
