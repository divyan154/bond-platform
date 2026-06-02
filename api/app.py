"""Flask REST API exposing every engine."""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import FLASK_HOST, FLASK_PORT
from database.sqlite_store    import fetch_all, fetch_by_isin, query
from database.knowledge_graph import issuer_exposure, sector_issuers, bond_neighbours
from engines.financial_engine import FinancialEngine
from engines.search_engine    import natural_language_search, filter_search
from engines.rag_engine       import ask as rag_ask
from engines.compliance_engine import ComplianceEngine
from engines.alerting_engine  import generate_alerts

app = Flask(__name__)
CORS(app)


def _ok(data, **extra):
    return jsonify({"ok": True, "data": data, **extra})


def _err(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


def _jsonable_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to JSON-safe records.

    Numeric NaN -> None (becomes null in JSON), preserving column dtypes
    so the UI can rebuild the DataFrame without Arrow serialization errors.
    String "nan"/"None" are also nulled.
    """
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    for row in records:
        for k, v in list(row.items()):
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
            elif isinstance(v, (pd.Timestamp,)):
                row[k] = v.isoformat()
            elif isinstance(v, str) and v.strip().lower() in ("nan", "none", ""):
                row[k] = None
    return records


@app.get("/health")
def health():
    return _ok({"status": "alive"})


@app.get("/api/bonds")
def list_bonds():
    limit  = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    df = fetch_all().iloc[offset:offset + limit]
    return _ok(_jsonable_records(df), total=int(len(fetch_all())))


@app.get("/api/bonds/<isin>")
def get_bond(isin):
    df = fetch_by_isin(isin)
    if df.empty:
        return _err(f"ISIN {isin} not found", 404)
    return _ok(_jsonable_records(df)[0])


@app.post("/api/search")
def search():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("query") or request.args.get("q") or "").strip()
    if not q:
        return _err("missing 'query'")
    ComplianceEngine.audit("api_user", "search", {"query": q})
    result = natural_language_search(q, k=int(body.get("k", 25)))
    if result.get("structured_rows"):
        result["structured_rows"] = _jsonable_records(
            pd.DataFrame(result["structured_rows"])
        )
    return _ok(result)


@app.post("/api/search/filter")
def search_filter():
    body = request.get_json(force=True, silent=True) or {}
    df = filter_search(**body)
    return _ok(_jsonable_records(df), total=int(len(df)))


@app.post("/api/analytics")
def analytics():
    body = request.get_json(force=True, silent=True) or {}
    isin = body.get("isin")
    row = body
    if isin:
        df = fetch_by_isin(isin)
        if df.empty:
            return _err(f"ISIN {isin} not found", 404)
        row = _jsonable_records(df)[0]
    bench = body.get("benchmark_yield")
    res = FinancialEngine.analyse(row, benchmark_yield=bench)
    ComplianceEngine.audit("api_user", "analytics", {"isin": isin})
    return _ok({"row": row, "analytics": res.as_dict()})


@app.post("/api/copilot")
def copilot():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("question") or "").strip()
    if not q:
        return _err("missing 'question'")
    out = rag_ask(q, k=int(body.get("k", 6)))
    ComplianceEngine.audit("api_user", "copilot", {"q": q,
                            "engine": out.get("engine")})
    return _ok(out)


@app.get("/api/kg/issuer/<name>")
def kg_issuer(name):
    return _ok(issuer_exposure(name))


@app.get("/api/kg/sector/<name>")
def kg_sector(name):
    return _ok({"sector": name, "issuers": sector_issuers(name)})


@app.get("/api/kg/bond/<isin>")
def kg_bond(isin):
    return _ok(bond_neighbours(isin))


@app.get("/api/alerts")
def alerts():
    args = request.args
    out = generate_alerts(
        yield_threshold=float(args.get("yield_threshold", 9.0)),
        maturity_days=int(args.get("maturity_days", 90)),
        price_dev=float(args.get("price_dev", 0.10)),
        rating_floor=args.get("rating_floor", "BBB"),
        min_volume=float(args.get("min_volume", 1000.0)),
    )
    return _ok(out, total=len(out))


@app.get("/api/audit")
def audit():
    return _ok(ComplianceEngine.recent_audit(int(request.args.get("n", 50))))


@app.get("/api/portfolio/summary")
def portfolio_summary():
    df = fetch_all()
    if df.empty:
        return _ok({})
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df["LastTradedPrice"] = pd.to_numeric(df["LastTradedPrice"], errors="coerce")
    df["FaceValue"]       = pd.to_numeric(df["FaceValue"], errors="coerce")
    summary = {
        "total_bonds":      int(len(df)),
        "unique_issuers":   int(df["IssuerName"].nunique()),
        "unique_sectors":   int(df["Sector"].nunique()),
        "rating_breakdown": df["Rating"].value_counts().to_dict(),
        "sector_breakdown": df["Sector"].value_counts().to_dict(),
        "country_breakdown": df["Country"].value_counts().to_dict(),
        "avg_yield":        float(df["Yield"].dropna().mean() or 0),
        "avg_price":        float(df["LastTradedPrice"].dropna().mean() or 0),
        "sources":          df["Source"].value_counts().to_dict(),
    }
    return _ok(summary)


def run():
    print(f"[api] http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    run()
