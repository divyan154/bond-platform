"""SQLite-backed structured store for the Bond Master."""
from __future__ import annotations
import sys
import sqlite3
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SQLITE_DB, UNIFIED_CSV

TABLE = "bonds"


def build_sqlite(df: pd.DataFrame | None = None) -> int:
    if df is None:
        df = pd.read_csv(UNIFIED_CSV)
    SQLITE_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SQLITE_DB) as con:
        df.to_sql(TABLE, con, if_exists="replace", index=False)
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_isin    ON {TABLE}(ISIN)")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_issuer  ON {TABLE}(IssuerName)")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_rating  ON {TABLE}(Rating)")
        con.execute(f"CREATE INDEX IF NOT EXISTS idx_sector  ON {TABLE}(Sector)")
        con.commit()
    print(f"[sqlite] wrote {len(df)} rows -> {SQLITE_DB.name}")
    return len(df)


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(SQLITE_DB) as con:
        return pd.read_sql_query(sql, con, params=params)


def fetch_all() -> pd.DataFrame:
    return query(f"SELECT * FROM {TABLE}")


def fetch_by_isin(isin: str) -> pd.DataFrame:
    return query(f"SELECT * FROM {TABLE} WHERE ISIN = ?", (isin.upper(),))


if __name__ == "__main__":
    build_sqlite()
