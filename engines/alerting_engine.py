"""
Alerting engine — scans the current Bond Master and produces alerts:

* YIELD_SPIKE     yield > yield_threshold
* MATURITY        maturity within `days` from today
* PRICE_OUTLIER   |price - face_value| > price_dev * face_value
* RATING_LOW      rating below floor
* LIQUIDITY_LOW   volume < min_volume
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from database.sqlite_store import fetch_all


def generate_alerts(yield_threshold: float = 9.0,
                    maturity_days:   int = 90,
                    price_dev:       float = 0.10,
                    rating_floor:    str = "BBB",
                    min_volume:      float = 1000.0) -> list[dict]:
    df = fetch_all()
    if df.empty:
        return []

    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"], errors="coerce")
    df["Yield"]        = pd.to_numeric(df["Yield"], errors="coerce")
    df["LastTradedPrice"] = pd.to_numeric(df["LastTradedPrice"], errors="coerce")
    df["FaceValue"]       = pd.to_numeric(df["FaceValue"], errors="coerce").fillna(1000)
    df["Volume"]          = pd.to_numeric(df["Volume"], errors="coerce")

    today = pd.Timestamp(date.today())
    horizon = pd.Timestamp(date.today() + timedelta(days=maturity_days))
    alerts: list[dict] = []

    for _, r in df.iterrows():
        isin = r["ISIN"]
        issuer = r.get("IssuerName") or r.get("Symbol")

        if pd.notna(r["Yield"]) and r["Yield"] >= yield_threshold:
            alerts.append({
                "type": "YIELD_SPIKE", "severity": "HIGH",
                "ISIN": isin, "issuer": issuer,
                "message": f"Yield {r['Yield']:.2f}% breaches {yield_threshold}%",
            })

        if pd.notna(r["MaturityDate"]) and today <= r["MaturityDate"] <= horizon:
            days = (r["MaturityDate"] - today).days
            alerts.append({
                "type": "MATURITY", "severity": "MEDIUM",
                "ISIN": isin, "issuer": issuer,
                "message": f"Matures in {days} days "
                           f"({r['MaturityDate'].date()})",
            })

        if pd.notna(r["LastTradedPrice"]):
            dev = abs(r["LastTradedPrice"] - r["FaceValue"]) / r["FaceValue"]
            if dev > price_dev:
                alerts.append({
                    "type": "PRICE_OUTLIER", "severity": "LOW",
                    "ISIN": isin, "issuer": issuer,
                    "message": f"Price {r['LastTradedPrice']} deviates "
                               f"{dev*100:.1f}% from face value",
                })

        rating = str(r.get("Rating") or "").upper()
        ladder = ["D", "C", "B", "BB", "BBB", "A", "AA", "AAA"]
        floor_idx = ladder.index(rating_floor.upper())
        base = next((g for g in ladder if rating.startswith(g)), None)
        if base and ladder.index(base) < floor_idx:
            alerts.append({
                "type": "RATING_LOW", "severity": "HIGH",
                "ISIN": isin, "issuer": issuer,
                "message": f"Rating {rating} below floor {rating_floor}",
            })

        if pd.notna(r["Volume"]) and r["Volume"] < min_volume:
            alerts.append({
                "type": "LIQUIDITY_LOW", "severity": "MEDIUM",
                "ISIN": isin, "issuer": issuer,
                "message": f"Volume {int(r['Volume'])} below threshold "
                           f"{int(min_volume)}",
            })

    return alerts


if __name__ == "__main__":
    for a in generate_alerts()[:10]:
        print(a)
