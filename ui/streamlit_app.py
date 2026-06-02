"""
Streamlit UI — AI-Powered Bond Intelligence & Execution Platform.

Pages:
    Dashboard      — portfolio summary
    Search         — natural language + filter search
    Bond Detail    — analytics + KG neighbourhood
    AI Copilot     — RAG-grounded Q&A
    Knowledge Graph— issuer/sector exposure browser
    Alerts         — live alert generator
    Compliance     — audit log + suitability check
    Admin          — rebuild databases
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import API_BASE, COMPLIANCE_DISCLAIMER

NUMERIC_COLS = ("CouponRate", "FaceValue", "LastTradedPrice", "BidPrice",
                "AskPrice", "Volume", "Yield", "Duration", "Spread")
DATE_COLS = ("IssueDate", "MaturityDate")


def coerce_df(rows: list[dict]) -> pd.DataFrame:
    """Rebuild a DataFrame from API records with Arrow-safe dtypes."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

st.set_page_config(
    page_title="Bond Intelligence Platform",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=60)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


def api_post(path: str, payload: dict | None = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=120)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}


with st.sidebar:
    st.title("🏦 Bond Intelligence")
    st.caption("Zintellix Technologies")
    page = st.radio("Navigate", [
        "Dashboard", "Search", "Bond Detail", "AI Copilot",
        "Knowledge Graph", "Alerts", "Compliance", "Admin",
    ])
    st.divider()
    h = api_get("/health")
    if h.get("ok"):
        st.success("API: connected")
    else:
        st.error(f"API offline: {h.get('error','?')}")
        st.code(f"Start with: python -m api.app")
    st.divider()
    st.caption(COMPLIANCE_DISCLAIMER)


# -------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.header("📊 Platform Dashboard")
    s = api_get("/api/portfolio/summary")
    if not s.get("ok"):
        st.error(s.get("error")); st.stop()
    d = s["data"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Bonds",    d.get("total_bonds", 0))
    c2.metric("Unique Issuers", d.get("unique_issuers", 0))
    c3.metric("Avg Yield (%)",  f"{d.get('avg_yield', 0):.2f}")
    c4.metric("Avg Price",      f"{d.get('avg_price', 0):.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Rating Distribution")
        rd = d.get("rating_breakdown", {})
        if rd:
            st.bar_chart(pd.Series(rd, name="Bonds"))
        st.subheader("Source Distribution")
        sd = d.get("sources", {})
        if sd:
            st.bar_chart(pd.Series(sd, name="Bonds"))
    with c2:
        st.subheader("Sector Distribution")
        sx = d.get("sector_breakdown", {})
        if sx:
            st.bar_chart(pd.Series(sx, name="Bonds"))
        st.subheader("Country Distribution")
        co = d.get("country_breakdown", {})
        if co:
            st.bar_chart(pd.Series(co, name="Bonds"))

    st.subheader("Latest Bond Master (first 200)")
    bl = api_get("/api/bonds", limit=200)
    if bl.get("ok"):
        st.dataframe(coerce_df(bl["data"]), width="stretch", hide_index=True)


# -------------------------------------------------------------- Search
elif page == "Search":
    st.header("🔍 Real-Time Bond Search")
    tab1, tab2 = st.tabs(["Natural Language", "Advanced Filter"])

    with tab1:
        q = st.text_input("Ask the Bond Search Engine",
                          placeholder="AA bonds maturing within 3 years above 7% YTM")
        k = st.slider("Max results", 5, 100, 25)
        if st.button("Search", type="primary") and q:
            r = api_post("/api/search", {"query": q, "k": k})
            if not r.get("ok"):
                st.error(r.get("error")); st.stop()
            data = r["data"]
            st.info(f"Parsed filters: `{data['parsed_filters']}` — "
                    f"{data['total']} structured matches")
            if data["structured_rows"]:
                st.subheader("Structured matches")
                st.dataframe(coerce_df(data["structured_rows"]),
                             width="stretch", hide_index=True)
            if data["semantic_hits"]:
                st.subheader("Semantic matches (ChromaDB)")
                rows = []
                for h in data["semantic_hits"]:
                    m = h.get("metadata", {}) or {}
                    rows.append({
                        "ISIN": h.get("id"), "Issuer": m.get("IssuerName"),
                        "Sector": m.get("Sector"), "Rating": m.get("Rating"),
                        "Coupon": m.get("CouponRate"), "Score": h.get("score"),
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab2:
        with st.form("filter"):
            c1, c2, c3 = st.columns(3)
            rating = c1.text_input("Rating starts with", "")
            min_y  = c2.number_input("Min yield (%)", value=0.0)
            max_y  = c3.number_input("Max yield (%)", value=0.0)
            c1, c2, c3 = st.columns(3)
            sector = c1.text_input("Sector contains", "")
            order  = c2.selectbox("Sort", ["", "highest", "lowest"])
            limit  = c3.number_input("Limit", value=50, min_value=1)
            submitted = st.form_submit_button("Run filter")
        if submitted:
            payload = {"rating": rating or None,
                       "min_yield": min_y or None,
                       "max_yield": max_y or None,
                       "sector_like": sector or None,
                       "order": order or None,
                       "limit": int(limit)}
            r = api_post("/api/search/filter",
                         {k: v for k, v in payload.items() if v is not None})
            if r.get("ok"):
                st.success(f"{r.get('total',0)} matches")
                st.dataframe(coerce_df(r["data"]),
                             width="stretch", hide_index=True)
            else:
                st.error(r.get("error"))


# -------------------------------------------------------------- Bond Detail
elif page == "Bond Detail":
    st.header("📑 Bond Detail & Analytics")
    bl = api_get("/api/bonds", limit=500)
    if not bl.get("ok"):
        st.error(bl.get("error")); st.stop()
    bonds = coerce_df(bl["data"])
    if bonds.empty:
        st.warning("No bonds in database. Run the pipeline first.")
        st.stop()
    options = bonds.apply(
        lambda r: f"{r['ISIN']} — {r.get('IssuerName') or r.get('Symbol') or 'Unknown'}",
        axis=1,
    ).tolist()
    pick = st.selectbox("Choose a bond", options)
    isin = pick.split(" — ")[0].strip()

    detail = api_get(f"/api/bonds/{isin}")
    if not detail.get("ok"):
        st.error(detail.get("error")); st.stop()
    row = detail["data"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Metadata")
        st.json({k: v for k, v in row.items() if v not in ("", None)})

    with c2:
        st.subheader("Run Analytics")
        bench = st.number_input("Benchmark yield (decimal, e.g. 0.07)",
                                value=0.07, step=0.01, format="%.4f")
        if st.button("Compute YTM, Duration, Convexity", type="primary"):
            res = api_post("/api/analytics",
                           {"isin": isin, "benchmark_yield": bench})
            if res.get("ok"):
                an = res["data"]["analytics"]
                m = st.columns(3)
                m[0].metric("YTM",         f"{(an.get('ytm') or 0)*100:.3f}%")
                m[1].metric("Mod Duration", f"{an.get('modified_duration') or 0:.3f}")
                m[2].metric("Convexity",    f"{an.get('convexity') or 0:.3f}")
                m = st.columns(3)
                m[0].metric("Clean Price",  f"{an.get('clean_price') or 0:.2f}")
                m[1].metric("Dirty Price",  f"{an.get('dirty_price') or 0:.2f}")
                m[2].metric("Spread (bps)", f"{an.get('spread_bps') or 0:.1f}")
                if an.get("notes"):
                    st.warning(" | ".join(an["notes"]))
            else:
                st.error(res.get("error"))

    st.subheader("Knowledge Graph neighbours")
    nbrs = api_get(f"/api/kg/bond/{isin}")
    if nbrs.get("ok"):
        st.json(nbrs["data"])


# -------------------------------------------------------------- Copilot
elif page == "AI Copilot":
    st.header("🤖 AI Bond Copilot (RAG)")
    st.caption("Retrieval-grounded. Cites verified ISINs. SEBI-aligned guardrails.")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("Ask anything about the bond corpus..."):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and grounding answer ..."):
                r = api_post("/api/copilot", {"question": q, "k": 6})
            if not r.get("ok"):
                st.error(r.get("error"))
            else:
                d = r["data"]
                st.markdown(d["answer"])
                col1, col2 = st.columns(2)
                col1.metric("Confidence", f"{d['confidence']:.2f}")
                col2.metric("Engine", d["engine"])
                if d.get("blocked"):
                    st.warning("Response blocked: low confidence.")
                if d.get("compliance_flags"):
                    st.error(f"Compliance scrub: {d['compliance_flags']}")
                with st.expander("Cited ISINs & evidence"):
                    st.write(d.get("citations"))
                    st.json(d.get("evidence"))
                st.session_state.chat.append(
                    {"role": "assistant", "content": d["answer"]})


# -------------------------------------------------------------- KG
elif page == "Knowledge Graph":
    st.header("🕸️ Knowledge Graph Browser")
    tab1, tab2 = st.tabs(["By Issuer", "By Sector"])
    with tab1:
        issuer = st.text_input("Issuer name", "HDFC Ltd")
        if st.button("Lookup issuer"):
            r = api_get(f"/api/kg/issuer/{issuer}")
            st.json(r.get("data"))
    with tab2:
        sector = st.text_input("Sector name", "Banking")
        if st.button("Lookup sector"):
            r = api_get(f"/api/kg/sector/{sector}")
            st.json(r.get("data"))


# -------------------------------------------------------------- Alerts
elif page == "Alerts":
    st.header("🚨 Alert Center")
    with st.form("alerts"):
        c = st.columns(5)
        yt = c[0].number_input("Yield threshold (%)", value=9.0)
        md = c[1].number_input("Maturity within (days)", value=90)
        pd_ = c[2].number_input("Price dev frac",
                                value=0.10, format="%.2f")
        rf = c[3].text_input("Rating floor", "BBB")
        mv = c[4].number_input("Min volume", value=1000.0)
        run = st.form_submit_button("Scan")
    if run:
        r = api_get("/api/alerts", yield_threshold=yt, maturity_days=md,
                    price_dev=pd_, rating_floor=rf, min_volume=mv)
        if r.get("ok"):
            df = pd.DataFrame(r["data"]).astype(str) if r["data"] else pd.DataFrame()
            st.success(f"{len(df)} alerts")
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.error(r.get("error"))


# -------------------------------------------------------------- Compliance
elif page == "Compliance":
    st.header("🛡️ Compliance & Audit")
    st.subheader("Recent audit log")
    r = api_get("/api/audit", n=50)
    if r.get("ok"):
        st.dataframe(pd.DataFrame(r["data"]),
                     width="stretch", hide_index=True)
    else:
        st.info("No audit entries yet.")
        st.write("(audit log appears after API calls run)")


# -------------------------------------------------------------- Admin
elif page == "Admin":
    st.header("⚙️ Admin Console")
    st.write("Rebuild any database from the latest CSVs.")
    if st.button("Run full pipeline (consolidate + rebuild ALL)"):
        with st.spinner("Running. Embedding ~all bonds may take a minute..."):
            try:
                from data_pipeline.data_consolidator import run as run_pipe
                from database.build_all import build_all
                run_pipe(try_live=True)
                build_all()
                st.success("Pipeline complete.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
